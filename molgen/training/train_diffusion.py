"""
train_diffusion.py

Phase 2 + 3 training: Graph Diffusion Model on QM9.

Phase 2 — Unconditional (run first, ~50 epochs):
    python -m molgen.training.train_diffusion --phase 2

Phase 3 — Conditional with CFG (fine-tune from Phase 2):
    python -m molgen.training.train_diffusion --phase 3 \
        --resume checkpoints/diffusion_phase2_best.pt

Kaggle/Colab:
    python -m molgen.training.train_diffusion --phase 2 \
        --data-root /content/data \
        --ckpt-dir  /content/drive/MyDrive/molgen_checkpoints

Target after Phase 3:
    Validity rate > 70% on sampled molecules
    FCD < 5.0
"""

import argparse
import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from molgen.data.dataset import get_dataloaders
from molgen.models.conditioning import PropertyConditioner
from molgen.models.denoiser import GraphDenoiser
from molgen.utils.config import load_diffusion_config
from molgen.utils.noise_schedule import CosineNoiseSchedule


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU — using CPU. Use --max-mols 500 for local testing.")
    return device


# ── Loss ───────────────────────────────────────────────────────────────────────

def diffusion_loss(
    eps_pred:    torch.Tensor,
    eps_true:    torch.Tensor,
    bond_logits: torch.Tensor,
    bond_true:   torch.Tensor,
    lambda_edge: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combined diffusion loss:
        L_node = MSE(predicted noise, actual noise)       — node features
        L_edge = BCE(bond logits, true bond existence)    — adjacency

    Args:
        eps_pred:    Predicted noise on node features  [N, 18]
        eps_true:    Actual noise that was added       [N, 18]
        bond_logits: Raw bond predictions              [E, 1]
        bond_true:   Ground truth bond existence       [E, 1]  (0 or 1)
        lambda_edge: Weight for edge loss (default 1.0)

    Returns:
        (total_loss, {'node_loss': float, 'edge_loss': float})
    """
    node_loss = nn.functional.mse_loss(eps_pred, eps_true)
    edge_loss = nn.functional.binary_cross_entropy_with_logits(
        bond_logits, bond_true
    )
    total = node_loss + lambda_edge * edge_loss

    return total, {
        "node_loss": float(node_loss.item()),
        "edge_loss": float(edge_loss.item()),
    }


# ── Build adjacency ground truth for a batch ───────────────────────────────────

def build_bond_targets(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    For every directed edge in edge_index, the bond exists (label = 1).
    Returns a [E, 1] tensor of ones — all edges in our graph are real bonds.

    During the forward process we corrupt the adjacency by flipping bits.
    The denoiser is trained to predict the clean adjacency (all 1s for
    real edges). The edge_index itself already encodes which pairs are bonded.

    Args:
        edge_index: [2, E] — existing bonds (both directions)
        num_nodes:  Total number of nodes in the batch

    Returns:
        bond_true: [E, 1] — all ones (these edges are real)
    """
    E = edge_index.shape[1]
    return torch.ones(E, 1, device=edge_index.device)


# ── One epoch ──────────────────────────────────────────────────────────────────

def run_epoch(
    denoiser:    GraphDenoiser,
    conditioner: PropertyConditioner,
    schedule:    CosineNoiseSchedule,
    loader,
    optimizer:   Optional[torch.optim.Optimizer],
    device:      torch.device,
    is_train:    bool,
    phase:       int,
    lambda_edge: float,
    T:           int,
) -> tuple[float, dict]:
    """
    Run one full pass over the dataset.

    For each batch:
    1. Sample random timestep t for each graph
    2. Add noise to node features via q_sample
    3. Get conditioning embedding (null for phase 2, real for phase 3)
    4. Run denoiser to predict noise + bond logits
    5. Compute loss and backprop

    Args:
        phase: 2 = unconditional (c always null), 3 = conditional (CFG)
    """
    if is_train:
        denoiser.train()
        conditioner.train()
    else:
        denoiser.eval()
        conditioner.eval()

    total_loss      = 0.0
    total_node_loss = 0.0
    total_edge_loss = 0.0
    n_batches       = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            batch = batch.to(device)
            B     = batch.num_graphs
            N     = batch.x.shape[0]

            # ── Sample timestep per graph, broadcast to nodes ──────────────
            # Each graph in the batch gets its own random t
            t_graph = torch.randint(1, T + 1, (B,), device=device)  # [B]
            t_node  = t_graph[batch.batch]                           # [N]

            # ── Forward process: corrupt node features ─────────────────────
            x_t, eps_true = schedule.q_sample(batch.x, t_node)      # [N,18]

            # ── Conditioning ───────────────────────────────────────────────
            if phase == 2:
                # Phase 2: unconditional — always use null conditioning
                props = torch.zeros(B, 4, device=device)
                c     = conditioner(props, force_null=True)           # [B, 64]
            else:
                # Phase 3: conditional with CFG dropout (handled inside conditioner)
                props = batch.y.view(B, 4)                           # [B, 4]
                c     = conditioner(props)                            # [B, 64]

            # ── Denoiser forward pass ──────────────────────────────────────
            eps_pred, bond_logits = denoiser(
                x_t,
                batch.edge_index,
                batch.edge_attr,
                t_node,
                c,
                batch.batch,
            )

            # ── Build edge targets (all existing edges are real bonds) ─────
            bond_true = build_bond_targets(batch.edge_index, N)

            # ── Loss ───────────────────────────────────────────────────────
            loss, loss_components = diffusion_loss(
                eps_pred, eps_true,
                bond_logits, bond_true,
                lambda_edge=lambda_edge,
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(denoiser.parameters()) + list(conditioner.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

            total_loss      += float(loss.item())
            total_node_loss += loss_components["node_loss"]
            total_edge_loss += loss_components["edge_loss"]
            n_batches       += 1

    n = max(n_batches, 1)
    return float(total_loss / n), {
        "node_loss": total_node_loss / n,
        "edge_loss": total_edge_loss / n,
    }


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(
    path:        str,
    denoiser:    GraphDenoiser,
    conditioner: PropertyConditioner,
    optimizer:   torch.optim.Optimizer,
    epoch:       int,
    best_val:    float,
    history:     dict,
    phase:       int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":             epoch,
        "phase":             phase,
        "denoiser_state":    denoiser.state_dict(),
        "conditioner_state": conditioner.state_dict(),
        "optimizer_state":   optimizer.state_dict(),
        "best_val_loss":     best_val,
        "history":           history,
    }, path)


def load_checkpoint(
    path:        str,
    denoiser:    GraphDenoiser,
    conditioner: PropertyConditioner,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
) -> tuple[int, float, dict, int]:
    print(f"Resuming from: {path}")
    ckpt = torch.load(path, map_location=device)
    denoiser.load_state_dict(ckpt["denoiser_state"])
    conditioner.load_state_dict(ckpt["conditioner_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return (
        ckpt["epoch"] + 1,
        ckpt["best_val_loss"],
        ckpt["history"],
        ckpt.get("phase", 2),
    )


# ── Main training loop ─────────────────────────────────────────────────────────

def train(
    config_path:    str = "configs/diffusion_config.json",
    resume_path:    Optional[str] = None,
    phase:          int = 2,
    max_mols:       Optional[int] = None,
    data_root:      str = "data",
    checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Full diffusion training loop — Phase 2 or Phase 3.

    Phase 2: Train unconditionally. Validates the denoising works.
             Target: node loss < 0.5, edge loss < 0.3 after 50 epochs.

    Phase 3: Fine-tune from Phase 2 with conditioning + CFG.
             Target: validity > 70% on sampled molecules.
    """
    cfg    = load_diffusion_config(config_path)
    device = get_device()
    T      = cfg["T"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\nPhase {phase} diffusion training")
    print(f"Config: T={T}, hidden={cfg['hidden_dim']}, "
          f"layers={cfg['num_layers']}, lr={cfg['learning_rate']}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, val_loader, _ = get_dataloaders(
        root=data_root,
        batch_size=cfg["batch_size"],
        max_mols=max_mols,
    )
    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # ── Models ────────────────────────────────────────────────────────────────
    denoiser = GraphDenoiser(
        node_dim=18,
        edge_dim=6,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        time_emb_dim=cfg["time_emb_dim"],
        cond_dim=cfg["conditioning_dim"],
    ).to(device)

    conditioner = PropertyConditioner(
        num_properties=4,
        conditioning_dim=cfg["conditioning_dim"],
        null_prob=cfg["null_conditioning_prob"],
    ).to(device)

    n_params = (
        sum(p.numel() for p in denoiser.parameters()) +
        sum(p.numel() for p in conditioner.parameters())
    )
    print(f"Total parameters: {n_params:,}")

    # ── Noise schedule ────────────────────────────────────────────────────────
    schedule = CosineNoiseSchedule(T=T).to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    all_params = list(denoiser.parameters()) + list(conditioner.parameters())
    optimizer  = Adam(all_params, lr=cfg["learning_rate"])
    scheduler  = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch    = 0
    best_val       = float("inf")
    patience_count = 0
    early_stop     = 20
    history        = {"train_loss": [], "val_loss": [],
                      "train_node": [], "train_edge": []}

    if resume_path and os.path.exists(resume_path):
        start_epoch, best_val, history, loaded_phase = load_checkpoint(
            resume_path, denoiser, conditioner, optimizer, device
        )
        print(f"Resumed epoch {start_epoch}, phase {loaded_phase}, "
              f"best val {best_val:.4f}")
        
    
    if resume_path and os.path.exists(resume_path):
        start_epoch, best_val, history, loaded_phase = load_checkpoint(
            resume_path, denoiser, conditioner, optimizer, device
        )
        # Reset best_val if switching phases so Phase 3 can save its own checkpoint
        if loaded_phase != phase:
            best_val = float("inf")
            print(f"Phase changed {loaded_phase}→{phase}: resetting best val loss.")
        print(f"Resumed epoch {start_epoch}, best val {best_val:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting from epoch {start_epoch + 1}/{cfg['epochs']}")
    print("-" * 70)

    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()

        train_loss, train_comps = run_epoch(
            denoiser, conditioner, schedule,
            train_loader, optimizer, device,
            is_train=True, phase=phase,
            lambda_edge=cfg["lambda_edge"], T=T,
        )
        val_loss, val_comps = run_epoch(
            denoiser, conditioner, schedule,
            val_loader, None, device,
            is_train=False, phase=phase,
            lambda_edge=cfg["lambda_edge"], T=T,
        )

        elapsed = time.time() - t0
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_node"].append(train_comps["node_loss"])
        history["train_edge"].append(train_comps["edge_loss"])

        print(
            f"Epoch {epoch+1:03d}/{cfg['epochs']} | "
            f"train: {train_loss:.4f} "
            f"(node {train_comps['node_loss']:.3f} "
            f"edge {train_comps['edge_loss']:.3f}) | "
            f"val: {val_loss:.4f} | {elapsed:.1f}s"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt_name = f"diffusion_phase{phase}_best.pt"
        if val_loss < best_val:
            best_val       = val_loss
            patience_count = 0
            save_checkpoint(
                os.path.join(checkpoint_dir, ckpt_name),
                denoiser, conditioner, optimizer,
                epoch, best_val, history, phase,
            )
            print(f"  ✓ Best val: {best_val:.4f} — saved")
        else:
            patience_count += 1

        if (epoch + 1) % cfg["checkpoint_every"] == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir,
                             f"diffusion_phase{phase}_epoch{epoch+1}.pt"),
                denoiser, conditioner, optimizer,
                epoch, best_val, history, phase,
            )

        if patience_count >= early_stop:
            print(f"\nEarly stopping after {early_stop} epochs without improvement.")
            break

    # ── Save history ───────────────────────────────────────────────────────────
    hist_path = os.path.join(checkpoint_dir, f"diffusion_phase{phase}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"History: {hist_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MolGen diffusion model")
    parser.add_argument("--config",    default="configs/diffusion_config.json")
    parser.add_argument("--resume",    default=None)
    parser.add_argument("--phase",     type=int, default=2, choices=[2, 3],
                        help="2=unconditional, 3=conditional")
    parser.add_argument("--max-mols",  type=int, default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ckpt-dir",  default="checkpoints")
    args = parser.parse_args()

    train(
        config_path=args.config,
        resume_path=args.resume,
        phase=args.phase,
        max_mols=args.max_mols,
        data_root=args.data_root,
        checkpoint_dir=args.ckpt_dir,
    )


if __name__ == "__main__":
    main()