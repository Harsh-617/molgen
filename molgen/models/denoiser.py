"""
denoiser.py

GNN denoiser for the graph diffusion model.

At each reverse diffusion step, the denoiser takes:
    - Noisy node features x_t        [N, node_dim]
    - Noisy adjacency (via edge_index)
    - Timestep t                      [N] (per node)
    - Conditioning embedding c        [B, cond_dim]
    - Graph batch vector              [N]

And predicts:
    - Noise on node features          [N, node_dim]
    - Bond existence probabilities    [E, 1]  (sigmoid applied)

Architecture:
    1. Input projection: [x_t || time_emb || c_broadcast] → hidden
    2. L=6 rounds of MPNN message passing (same as encoder but deeper)
    3. Node output head: hidden → node_dim  (predicted noise)
    4. Edge output head: MLP(h_u, h_v) → 1  (bond probability)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_add_pool


# ── Sinusoidal time embedding ──────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Encodes scalar timestep t into a fixed sinusoidal embedding,
    then projects it to time_emb_dim via a small MLP.

    Same technique as used in DDPM and Stable Diffusion.
    Gives the model a sense of "how noisy is this input right now".

    Args:
        time_emb_dim: Output dimension (64)
    """

    def __init__(self, time_emb_dim: int = 64) -> None:
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Small MLP to project raw sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Timestep per node   [N]  values in [0, T]

        Returns:
            Time embeddings        [N, time_emb_dim]
        """
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)   # [N, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)     # [N, time_emb_dim]
        return self.mlp(emb)


# ── Message Passing Layer (same as encoder but with time/conditioning input) ───

class DenoiserMPNNLayer(MessagePassing):
    """
    One round of message passing for the denoiser.

    Identical to the encoder's MPNNLayer but the input node features
    already include the time embedding and conditioning — no changes
    needed to the message or update logic.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1) -> None:
        super().__init__(aggr="add")

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        aggregated = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return self.gru(input=aggregated, hx=h)

    def message(self, h_i: Tensor, h_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.msg_mlp(torch.cat([h_j, h_i, edge_attr], dim=-1))


# ── Full Denoiser ──────────────────────────────────────────────────────────────

class GraphDenoiser(nn.Module):
    """
    GNN denoiser: given noisy graph at timestep t + conditioning c,
    predict the noise on node features and bond existence probabilities.

    Args:
        node_dim:       Raw node feature dimension (18)
        edge_dim:       Edge feature dimension (6)
        hidden_dim:     Hidden dimension throughout (128)
        num_layers:     Number of message passing rounds (6)
        time_emb_dim:   Time embedding dimension (64)
        cond_dim:       Conditioning embedding dimension (64)
        dropout:        Dropout probability (0.1)
    """

    def __init__(
        self,
        node_dim:     int = 18,
        edge_dim:     int = 6,
        hidden_dim:   int = 128,
        num_layers:   int = 6,
        time_emb_dim: int = 64,
        cond_dim:     int = 64,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        # Input projection: concatenate [x_t | time_emb | c] → hidden
        # x_t is node_dim, time_emb is time_emb_dim, c is cond_dim
        input_dim = node_dim + time_emb_dim + cond_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )

        # Message passing layers (6 — deeper than encoder's 4)
        self.mp_layers = nn.ModuleList([
            DenoiserMPNNLayer(hidden_dim, edge_dim, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Node output head: predict noise on node features
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),   # same dim as input features
        )

        # Edge output head: predict bond existence probability
        # Takes concatenated embeddings of both endpoint nodes
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),          # one logit per node pair
        )

    def forward(
        self,
        x_t:        Tensor,
        edge_index:  Tensor,
        edge_attr:   Tensor,
        t:           Tensor,
        c:           Tensor,
        batch:       Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x_t:        Noisy node features         [N, node_dim]
            edge_index: Bond connectivity            [2, E]
            edge_attr:  Bond features                [E, edge_dim]
            t:          Timestep per node            [N]
            c:          Conditioning per graph       [B, cond_dim]
            batch:      Graph membership per node    [N]

        Returns:
            eps_pred:   Predicted noise on nodes     [N, node_dim]
            bond_logits: Bond existence logits       [E, 1]
                         Apply sigmoid for probabilities.
        """
        N = x_t.shape[0]

        # ── Time embedding (per node) ──────────────────────────────────────
        t_emb = self.time_emb(t)             # [N, time_emb_dim]

        # ── Broadcast conditioning from graph-level to node-level ──────────
        # batch[i] tells us which graph node i belongs to
        c_node = c[batch]                    # [N, cond_dim]

        # ── Input projection ───────────────────────────────────────────────
        h = self.input_proj(
            torch.cat([x_t, t_emb, c_node], dim=-1)
        )                                    # [N, hidden_dim]

        # ── Message passing ────────────────────────────────────────────────
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr)
            h = self.dropout(h)

        # ── Node output: predicted noise ───────────────────────────────────
        eps_pred = self.node_head(h)         # [N, node_dim]

        # ── Edge output: bond existence logits ─────────────────────────────
        # For each edge (i→j), concatenate embeddings of both endpoints
        src, dst    = edge_index[0], edge_index[1]
        edge_input  = torch.cat([h[src], h[dst]], dim=-1)   # [E, 2*hidden]
        bond_logits = self.edge_head(edge_input)             # [E, 1]

        return eps_pred, bond_logits