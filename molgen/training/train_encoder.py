"""
train_encoder.py

Phase 1 training: MPNN property predictor on QM9.

Usage:
    python -m molgen.training.train_encoder
    python -m molgen.training.train_encoder --resume checkpoints/encoder_best.pt
    python -m molgen.training.train_encoder --max-mols 1000  # fast local test

Target: MAE < 0.3 on all four properties on validation set.
Full training (~100 epochs on full QM9): 2-4 hours on Kaggle T4.
Local testing with --max-mols 1000: ~5 minutes on CPU.
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
from molgen.models.encoder import MPNNEncoder
from molgen.utils.config import load_encoder_config


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU found — using CPU. Training will be slow on full QM9.")
        print("Use --max-mols 1000 for local testing, run full training on Kaggle.")
    return device


# ── Metrics ────────────────────────────────────────────────────────────────────

PROPERTY_NAMES = ["logp", "qed", "sa_score", "homo_lumo"]


def compute_loss_and_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute MSE loss (for backprop) and per-property MAE (for reporting).

    Args:
        predictions: [B, 4] — model output
        targets:     [B, 4] — normalised ground truth

    Returns:
        (total_loss, mae_dict)
        total_loss: sum of MSE across all 4 properties
        mae_dict:   {'logp': float, 'qed': float, 'sa_score': float, 'homo_lumo': float}
    """
    # MSE per property, then sum — all properties equally weighted
    mse_per_prop = ((predictions - targets) ** 2).mean(dim=0)  # [4]
    total_loss   = mse_per_prop.sum()

    # MAE per property for human-readable reporting
    mae_per_prop = (predictions - targets).abs().mean(dim=0)   # [4]
    mae_dict = {
        name: float(mae_per_prop[i].item())
        for i, name in enumerate(PROPERTY_NAMES)
    }

    return total_loss, mae_dict


# ── One epoch ──────────────────────────────────────────────────────────────────

def run_epoch(
    model:      MPNNEncoder,
    loader,
    optimizer:  Optional[torch.optim.Optimizer],
    device:     torch.device,
    is_train:   bool,
) -> tuple[float, dict[str, float]]:
    """
    Run one full pass over a DataLoader.

    Args:
        model:    The encoder.
        loader:   Train or val DataLoader.
        optimizer: Pass None for validation (no gradient update).
        device:   torch.device.
        is_train: If True, model.train() and backprop. If False, model.eval().

    Returns:
        (avg_loss, avg_mae_dict)
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mae  = {name: 0.0 for name in PROPERTY_NAMES}
    n_batches  = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            predictions = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )  # [B, 4]

            # y shape from DataLoader is [B, 4] after batching
            targets = batch.y.view(-1, 4)  # ensure [B, 4]

            loss, mae = compute_loss_and_mae(predictions, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — prevents exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(loss.item())
            for name in PROPERTY_NAMES:
                total_mae[name] += mae[name]
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_mae  = {name: total_mae[name] / max(n_batches, 1)
                for name in PROPERTY_NAMES}

    return avg_loss, avg_mae


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(
    path:      str,
    model:     MPNNEncoder,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    best_val_loss: float,
    history:   dict,
) -> None:
    """Save full training state so we can resume exactly."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "best_val_loss":  best_val_loss,
        "history":        history,
    }, path)


def load_checkpoint(
    path:      str,
    model:     MPNNEncoder,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> tuple[int, float, dict]:
    """
    Load training state from checkpoint.
    Returns (start_epoch, best_val_loss, history).
    """
    print(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"] + 1, ckpt["best_val_loss"], ckpt["history"]


# ── Main training loop ─────────────────────────────────────────────────────────

def train(
    config_path:  str = "configs/encoder_config.json",
    resume_path:  Optional[str] = None,
    max_mols:     Optional[int] = None,
    data_root:    str = "data",
    checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Full Phase 1 training loop.

    Args:
        config_path:    Path to encoder_config.json.
        resume_path:    Path to checkpoint to resume from (optional).
        max_mols:       Cap on molecules — use 1000 for local CPU testing.
        data_root:      Root directory for data.
        checkpoint_dir: Where to save checkpoints.
    """
    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_encoder_config(config_path)
    print(f"\nEncoder config: {cfg}")

    device = get_device()
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, val_loader, _ = get_dataloaders(
        root=data_root,
        batch_size=cfg["batch_size"],
        max_mols=max_mols,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MPNNEncoder(
        node_dim=18,
        edge_dim=6,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    # Halve LR if val loss plateaus for 10 epochs
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # ── Resume from checkpoint if requested ───────────────────────────────────
    start_epoch    = 0
    best_val_loss  = float("inf")
    patience_count = 0
    early_stop_patience = 15
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    if resume_path and os.path.exists(resume_path):
        start_epoch, best_val_loss, history = load_checkpoint(
            resume_path, model, optimizer, device
        )
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training from epoch {start_epoch + 1}/{cfg['epochs']}")
    print("-" * 70)

    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()

        train_loss, train_mae = run_epoch(
            model, train_loader, optimizer, device, is_train=True
        )
        val_loss, val_mae = run_epoch(
            model, val_loader, None, device, is_train=False
        )

        elapsed = time.time() - t0
        scheduler.step(val_loss)

        # ── Log ───────────────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        mae_str = " | ".join(
            f"{name}: {val_mae[name]:.3f}" for name in PROPERTY_NAMES
        )
        print(
            f"Epoch {epoch+1:03d}/{cfg['epochs']} | "
            f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
            f"MAE → {mae_str} | {elapsed:.1f}s"
        )

        # ── Save best checkpoint ───────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_checkpoint(
                os.path.join(checkpoint_dir, "encoder_best.pt"),
                model, optimizer, epoch, best_val_loss, history,
            )
            print(f"  ✓ New best val loss: {best_val_loss:.4f} — checkpoint saved")
        else:
            patience_count += 1

        # ── Periodic checkpoint ────────────────────────────────────────────────
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, f"encoder_epoch{epoch+1}.pt"),
                model, optimizer, epoch, best_val_loss, history,
            )

        # ── Early stopping ─────────────────────────────────────────────────────
        if patience_count >= early_stop_patience:
            print(f"\nEarly stopping: val loss hasn't improved for "
                  f"{early_stop_patience} epochs.")
            break

    # ── Save final history ────────────────────────────────────────────────────
    history_path = os.path.join(checkpoint_dir, "encoder_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {history_path}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {os.path.join(checkpoint_dir, 'encoder_best.pt')}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MolGen encoder (Phase 1)")
    parser.add_argument("--config",     default="configs/encoder_config.json")
    parser.add_argument("--resume",     default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max-mols",   type=int, default=None,
                        help="Cap on molecules (e.g. 1000 for fast local test)")
    parser.add_argument("--data-root",  default="data")
    parser.add_argument("--ckpt-dir",   default="checkpoints")
    args = parser.parse_args()

    train(
        config_path=args.config,
        resume_path=args.resume,
        max_mols=args.max_mols,
        data_root=args.data_root,
        checkpoint_dir=args.ckpt_dir,
    )


if __name__ == "__main__":
    main()