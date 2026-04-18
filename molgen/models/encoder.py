"""
encoder.py

MPNN property predictor — the backbone of MolGen.

Architecture:
    1. Linear input projection:  18 → 128  (per atom)
    2. L rounds of message passing:
         message  = MLP([h_u ‖ h_v ‖ e_uv])   (128+128+6 → 256 → 128)
         aggregate = sum over neighbours
         update   = GRU(input=aggregated, hidden=h_v)
    3. Graph readout: sum(h_v) / sqrt(N)       → [128]
    4. Four property heads (MLP 128→64→1):
         LogP, QED (+ sigmoid), SA Score, HOMO-LUMO gap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool


# ── Message Passing Layer ──────────────────────────────────────────────────────

class MPNNLayer(MessagePassing):
    """
    One round of message passing with edge features and GRU update.

    For each node v:
        msg_{u→v}  = MLP_msg([h_u ‖ h_v ‖ e_{uv}])
        agg_v      = sum_{u ∈ N(v)} msg_{u→v}
        h_v_new    = GRU(input=agg_v, hidden=h_v)
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1) -> None:
        super().__init__(aggr="add")   # sum aggregation

        self.hidden_dim = hidden_dim

        # Message MLP: [h_u ‖ h_v ‖ e_uv] → hidden
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # GRU cell for node update
        # input_size  = hidden_dim  (aggregated messages)
        # hidden_size = hidden_dim  (current node state)
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Args:
            h:          Node features         [N, hidden_dim]
            edge_index: Bond connectivity     [2, 2E]
            edge_attr:  Bond features         [2E, edge_dim]

        Returns:
            Updated node features             [N, hidden_dim]
        """
        # propagate() calls message() then aggregate() automatically
        # We pass h as both x (for message computation) and the GRU hidden state
        aggregated = self.propagate(edge_index, h=h, edge_attr=edge_attr)

        # GRU update: memory = h_v_old, new input = aggregated neighbour messages
        h_new = self.gru(input=aggregated, hx=h)
        return h_new

    def message(self, h_i: Tensor, h_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Compute message from node j → node i along each edge.

        PyG naming convention:
            h_i = features of the TARGET node  (receiver)
            h_j = features of the SOURCE node  (sender)

        Args:
            h_i:       Target node features    [E, hidden_dim]
            h_j:       Source node features    [E, hidden_dim]
            edge_attr: Edge features           [E, edge_dim]

        Returns:
            Messages                           [E, hidden_dim]
        """
        # Concatenate: what the sender knows + what the receiver knows + bond type
        combined = torch.cat([h_j, h_i, edge_attr], dim=-1)  # [E, 2*hidden + edge_dim]
        return self.msg_mlp(combined)                          # [E, hidden_dim]


# ── Property Head ──────────────────────────────────────────────────────────────

class PropertyHead(nn.Module):
    """
    Small MLP that maps a graph-level vector to a single scalar property.
    128 → 64 → 1
    """

    def __init__(self, hidden_dim: int, use_sigmoid: bool = False) -> None:
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_G: Tensor) -> Tensor:
        out = self.mlp(h_G)           # [B, 1]
        if self.use_sigmoid:
            out = torch.sigmoid(out)  # constrain to [0, 1] for QED
        return out


# ── Full MPNN Encoder ──────────────────────────────────────────────────────────

class MPNNEncoder(nn.Module):
    """
    Full MPNN property predictor for molecular graphs.

    Given a batch of molecular graphs, predicts four properties:
        LogP, QED, SA Score, HOMO-LUMO gap

    Args:
        node_dim:   Input node feature dimension (18 from our featurisation)
        edge_dim:   Input edge feature dimension (6)
        hidden_dim: Hidden dimension throughout the network (128)
        num_layers: Number of message passing rounds (4)
        dropout:    Dropout probability (0.1)
    """

    def __init__(
        self,
        node_dim:   int = 18,
        edge_dim:   int = 6,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project raw atom features into hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
        )

        # Stack of message passing layers
        self.mp_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_dim, dropout)
            for _ in range(num_layers)
        ])

        # One property head per target
        # QED uses sigmoid because it's always in [0, 1]
        self.head_logp     = PropertyHead(hidden_dim, use_sigmoid=False)
        self.head_qed      = PropertyHead(hidden_dim, use_sigmoid=False)
        self.head_sa       = PropertyHead(hidden_dim, use_sigmoid=False)
        self.head_homo_lumo= PropertyHead(hidden_dim, use_sigmoid=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Tensor, batch: Tensor) -> Tensor:
        """
        Args:
            x:          Node features         [N_total, 18]
            edge_index: Edge connectivity     [2, E_total]
            edge_attr:  Edge features         [E_total, 6]
            batch:      Graph membership      [N_total]
                        batch[i] = which graph node i belongs to

        Returns:
            Predicted properties              [B, 4]
            Columns: [logp, qed, sa_score, homo_lumo]
        """
        # ── Embed atoms ───────────────────────────────────────────────────────
        h = self.input_proj(x)          # [N, hidden_dim]

        # ── Message passing rounds ────────────────────────────────────────────
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr)
            h = self.dropout(h)

        # ── Graph readout: sum(h_v) / sqrt(N) ────────────────────────────────
        # global_add_pool sums all node vectors per graph → [B, hidden_dim]
        h_G = global_add_pool(h, batch)

        # Count atoms per graph for normalisation
        # batch.bincount() gives [B] tensor of atom counts
        n_atoms = batch.bincount().float().unsqueeze(-1)  # [B, 1]
        h_G = h_G / (n_atoms.sqrt() + 1e-8)              # [B, hidden_dim]

        # ── Property prediction ───────────────────────────────────────────────
        logp      = self.head_logp(h_G)      # [B, 1]
        qed       = self.head_qed(h_G)       # [B, 1]  — sigmoid applied inside
        sa        = self.head_sa(h_G)        # [B, 1]
        homo_lumo = self.head_homo_lumo(h_G) # [B, 1]

        return torch.cat([logp, qed, sa, homo_lumo], dim=-1)  # [B, 4]

    def encode(self, x: Tensor, edge_index: Tensor,
               edge_attr: Tensor, batch: Tensor) -> Tensor:
        """
        Return the graph-level embedding h_G without property heads.
        Used by the diffusion model's conditioning mechanism.

        Returns: [B, hidden_dim]
        """
        h = self.input_proj(x)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr)
        h_G = global_add_pool(h, batch)
        n_atoms = batch.bincount().float().unsqueeze(-1)
        return h_G / (n_atoms.sqrt() + 1e-8)