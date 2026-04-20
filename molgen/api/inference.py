"""
inference.py

Model loading and inference logic for the FastAPI backend.

Separates inference logic from routing so it can be
tested and used independently of FastAPI.
"""

from typing import Optional
import torch

from molgen.models.encoder import MPNNEncoder
from molgen.models.conditioning import PropertyConditioner
from molgen.models.denoiser import GraphDenoiser
from molgen.utils.noise_schedule import CosineNoiseSchedule
from molgen.utils.config import load_encoder_config, load_diffusion_config
from molgen.data.graph_builder import smiles_to_graph
from molgen.data.property_calculator import (
    compute_properties,
    PropertyNormaliser,
)
from molgen.evaluation.metrics import is_valid, canonicalise

from torch_geometric.data import Batch


class MolGenInference:
    """
    Loads trained models and provides generation + prediction methods.

    Args:
        encoder_ckpt_path:   Path to encoder_best.pt
        diffusion_ckpt_path: Path to diffusion_phase3_best.pt
        encoder_config:      Path to encoder_config.json
        diffusion_config:    Path to diffusion_config.json
    """

    def __init__(
        self,
        encoder_ckpt_path:   Optional[str] = None,
        diffusion_ckpt_path: Optional[str] = None,
        encoder_config:      str = "configs/encoder_config.json",
        diffusion_config:    str = "configs/diffusion_config.json",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inference device: {self.device}")

        enc_cfg  = load_encoder_config(encoder_config)
        diff_cfg = load_diffusion_config(diffusion_config)

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = MPNNEncoder(
            node_dim=18, edge_dim=6,
            hidden_dim=enc_cfg["hidden_dim"],
            num_layers=enc_cfg["num_layers"],
            dropout=enc_cfg["dropout"],
        ).to(self.device)

        if encoder_ckpt_path:
            self._load_encoder(encoder_ckpt_path)

        self.encoder.eval()

        # ── Diffusion model ───────────────────────────────────────────────────
        self.denoiser = GraphDenoiser(
            node_dim=18, edge_dim=6,
            hidden_dim=diff_cfg["hidden_dim"],
            num_layers=diff_cfg["num_layers"],
            time_emb_dim=diff_cfg["time_emb_dim"],
            cond_dim=diff_cfg["conditioning_dim"],
        ).to(self.device)

        self.conditioner = PropertyConditioner(
            num_properties=4,
            conditioning_dim=diff_cfg["conditioning_dim"],
            null_prob=0.0,    # no dropout at inference
        ).to(self.device)

        if diffusion_ckpt_path:
            self._load_diffusion(diffusion_ckpt_path)

        self.denoiser.eval()
        self.conditioner.eval()

        # ── Noise schedule ────────────────────────────────────────────────────
        self.T        = diff_cfg["T"]
        self.schedule = CosineNoiseSchedule(T=self.T).to(self.device)

        # ── Normaliser ────────────────────────────────────────────────────────
        self.normaliser = PropertyNormaliser()

    # ── Checkpoint loading ─────────────────────────────────────────────────────

    def _load_encoder(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.encoder.load_state_dict(ckpt["model_state"])
            print(f"Encoder loaded from {path}")
        except Exception as e:
            print(f"Warning: could not load encoder from {path}: {e}")

    def _load_diffusion(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.denoiser.load_state_dict(ckpt["denoiser_state"])
            self.conditioner.load_state_dict(ckpt["conditioner_state"])
            print(f"Diffusion model loaded from {path}")
        except Exception as e:
            print(f"Warning: could not load diffusion from {path}: {e}")

    # ── Property prediction ────────────────────────────────────────────────────

    def predict_properties(self, smiles: str) -> dict[str, float]:
        """
        Predict molecular properties for a SMILES string using the GNN encoder.
        Returns normalised predictions denormalised back to real units.
        """
        graph = smiles_to_graph(smiles)
        if graph is None:
            return {}

        batch = Batch.from_data_list([graph]).to(self.device)

        with torch.no_grad():
            pred = self.encoder(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )  # [1, 4] normalised

        pred = pred[0].cpu().tolist()   # [logp, qed, sa, homo_lumo] normalised

        # Denormalise
        normed = {
            "logp":      pred[0],
            "qed":       pred[1],
            "sa_score":  pred[2],
            "homo_lumo": pred[3],
        }
        return self.normaliser.denormalise(normed)

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate(
        self,
        target_properties: dict[str, float],
        n_molecules:       int = 10,
        guidance_scale:    float = 2.0,
        max_atoms:         int = 9,
    ) -> list[dict]:
        """
        Generate molecules conditioned on target properties.

        Uses DDPM reverse diffusion with classifier-free guidance.
        Returns a list of dicts with SMILES, validity, and predicted properties.

        Note: Full graph generation from random noise → valid SMILES requires
        converting the denoised node features + adjacency back to a molecular
        graph. This is implemented as a simplified version that generates
        node features and uses a threshold on bond logits for adjacency.

        Args:
            target_properties: Dict with keys logp, qed, sa_score, homo_lumo.
            n_molecules:       How many to attempt generating.
            guidance_scale:    CFG guidance scale w (higher = stronger conditioning).
            max_atoms:         Max atoms per molecule (QM9 = 9).

        Returns:
            List of result dicts.
        """
        # Normalise target properties
        normed = self.normaliser.normalise(target_properties)
        props_tensor = torch.tensor([[
            normed.get("logp",      0.0),
            normed.get("qed",       0.0),
            normed.get("sa_score",  0.0),
            normed.get("homo_lumo", 0.0),
        ]], dtype=torch.float, device=self.device)

        results = []

        with torch.no_grad():
            for _ in range(n_molecules):
                smiles = self._sample_one_molecule(
                    props_tensor, guidance_scale, max_atoms
                )
                valid  = smiles is not None and is_valid(smiles)

                result = {
                    "smiles": smiles or "",
                    "valid":  valid,
                }

                if valid:
                    try:
                        pred_props = self.predict_properties(smiles)
                        result["predicted_logp"]      = pred_props.get("logp")
                        result["predicted_qed"]       = pred_props.get("qed")
                        result["predicted_sa"]        = pred_props.get("sa_score")
                        result["predicted_homo_lumo"] = pred_props.get("homo_lumo")
                    except Exception:
                        # Generated molecule has degenerate structure
                        # (e.g. no bonds) — skip property prediction
                        result["valid"] = False

                results.append(result)

        return results

    def _sample_one_molecule(
        self,
        props_tensor:   torch.Tensor,
        guidance_scale: float,
        max_atoms:      int,
    ) -> Optional[str]:
        """
        Run the full reverse diffusion process to sample one molecule.

        This is a simplified graph generation: we denoise node features
        and predict bond existence, then reconstruct a molecule with RDKit.
        """
        from rdkit import Chem
        from rdkit.Chem import RWMol, Atom, BondType

        # Sample number of atoms (uniform over QM9 range)
        import random
        n_atoms = random.randint(3, max_atoms)

        # Build a minimal edge_index (fully connected graph for generation)
        # During generation we don't know which bonds exist yet
        pairs = [(i, j) for i in range(n_atoms) for j in range(n_atoms) if i != j]
        if not pairs:
            return None

        src = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=self.device)
        dst = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=self.device)
        edge_index = torch.stack([src, dst], dim=0)

        # Dummy edge features (single bonds — will be overridden by denoiser)
        E = edge_index.shape[1]
        edge_attr = torch.zeros(E, 6, device=self.device)
        edge_attr[:, 0] = 1.0   # single bond one-hot

        # Batch vector (all nodes belong to graph 0)
        batch = torch.zeros(n_atoms, dtype=torch.long, device=self.device)

        # Start from pure noise
        x_t = torch.randn(n_atoms, 18, device=self.device)

        # Conditioning embeddings
        c_cond = self.conditioner(props_tensor.expand(1, -1), force_null=False)
        c_null = self.conditioner(props_tensor.expand(1, -1), force_null=True)

        # Reverse diffusion loop
        for t_val in range(self.T, 0, -1):
            t_node = torch.full((n_atoms,), t_val, dtype=torch.long,
                                device=self.device)

            # Conditional prediction
            eps_cond, bond_logits_cond = self.denoiser(
                x_t, edge_index, edge_attr, t_node, c_cond, batch
            )

            # Unconditional prediction (for CFG)
            eps_uncond, _ = self.denoiser(
                x_t, edge_index, edge_attr, t_node, c_null, batch
            )

            # Classifier-free guidance
            eps = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond

            # DDPM step
            x_t = self.schedule.ddpm_step(x_t, t_val, eps)

        # Convert denoised features + bond logits to a molecule
        bond_probs = torch.sigmoid(bond_logits_cond).squeeze(-1)  # [E]

        try:
            return self._graph_to_smiles(
                x_t, edge_index, bond_probs, n_atoms
            )
        except Exception:
            return None

    def _graph_to_smiles(
        self,
        x:           torch.Tensor,
        edge_index:  torch.Tensor,
        bond_probs:  torch.Tensor,
        n_atoms:     int,
    ) -> Optional[str]:
        """
        Convert denoised node features + bond probabilities to a SMILES string.

        Decodes atom types from node features via argmax on the one-hot portion,
        then adds bonds where probability > 0.5.
        """
        from rdkit.Chem import RWMol, Atom
        from rdkit import Chem
        from molgen.data.graph_builder import ATOM_TYPES

        x_cpu = x.cpu()

        # Decode atom types from one-hot features (first 10 dims)
        atom_type_logits = x_cpu[:, :10]
        atom_type_idx    = atom_type_logits.argmax(dim=-1).tolist()

        mol = RWMol()
        for idx in atom_type_idx:
            symbol = ATOM_TYPES[idx] if idx < len(ATOM_TYPES) - 1 else "C"
            mol.AddAtom(Atom(symbol))

        # Add bonds where probability > 0.5 (undirected — only add once)
        added_bonds = set()
        src_list = edge_index[0].cpu().tolist()
        dst_list = edge_index[1].cpu().tolist()
        probs    = bond_probs.cpu().tolist()

        for s, d, p in zip(src_list, dst_list, probs):
            if s < d and p > 0.5:
                bond_key = (s, d)
                if bond_key not in added_bonds:
                    try:
                        mol.AddBond(s, d, Chem.rdchem.BondType.SINGLE)
                        added_bonds.add(bond_key)
                    except Exception:
                        pass

        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None