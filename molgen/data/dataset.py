"""
dataset.py

Loads QM9, builds molecular graphs with our featurisation,
attaches property labels, and provides train/val/test splits.

Downloads gdb9.tar.gz directly and parses with RDKit SDMolSupplier to
bypass PyG's built-in QM9 class, which crashes on malformed SDF entries
(AttributeError: 'NoneType' object has no attribute 'GetNumAtoms').

Uses a plain torch.utils.data.Dataset — no InMemoryDataset.
"""

import csv
import math
import os
import pickle
import tarfile
import urllib.request
from typing import Optional

import torch
from torch.utils.data import Dataset
from rdkit import Chem
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from molgen.data.graph_builder import smiles_to_graph
from molgen.data.property_calculator import compute_properties, PropertyNormaliser

# ── QM9 download constants ─────────────────────────────────────────────────────
QM9_URL       = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
SDF_FILENAME  = "gdb9.sdf"
CSV_FILENAME  = "gdb9.sdf.csv"

# Column layout of gdb9.sdf.csv (0-indexed):
#   0: mol_id | 1:A | 2:B | 3:C | 4:mu | 5:alpha | 6:homo | 7:lumo | 8:gap | ...
HOMO_LUMO_COL = 8


class MolGenDataset(Dataset):
    """
    QM9 dataset with our own graph featurisation and property labels.

    Each item is a PyG Data object with:
        x          [N, 18]   atom features
        edge_index [2, 2E]   bond connectivity
        edge_attr  [2E, 6]   bond features
        y          [1, 4]    z-score normalised [logp, qed, sa_score, homo_lumo]
        smiles     str       canonical SMILES

    Args:
        root:        Directory to store downloaded + cached data.
        split:       'train', 'val', or 'test'.
        normaliser:  Shared PropertyNormaliser. Stats are fitted on the train
                     split only — pass the same instance to all three splits.
        val_ratio:   Fraction for validation set  (default 0.10).
        test_ratio:  Fraction for test set        (default 0.10).
        seed:        Random seed for reproducible splits.
        max_mols:    Cap on total molecules processed (e.g. 50 for fast CI runs).
    """

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        normaliser: Optional[PropertyNormaliser] = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        max_mols: Optional[int] = None,
    ) -> None:
        assert split in ("train", "val", "test")
        self.root       = root
        self.split      = split
        self.normaliser = normaliser or PropertyNormaliser()
        self.val_ratio  = val_ratio
        self.test_ratio = test_ratio
        self.seed       = seed
        self.max_mols   = max_mols

        os.makedirs(root, exist_ok=True)
        self.data_list = self._load()

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _suffix(self) -> str:
        return f"_{self.max_mols}" if self.max_mols else "_full"

    def _cache_path(self) -> str:
        return os.path.join(self.root, f"molgen_{self.split}{self._suffix()}.pkl")

    def _normaliser_cache_path(self) -> str:
        return os.path.join(self.root, f"molgen_normaliser{self._suffix()}.pkl")

    # ── Load or build ──────────────────────────────────────────────────────────

    def _load(self) -> list:
        if os.path.exists(self._cache_path()):
            print(f"Loading cached {self.split} split from {self._cache_path()}")
            # Restore normaliser stats so denormalisation works after cache hits
            norm_path = self._normaliser_cache_path()
            if os.path.exists(norm_path):
                with open(norm_path, "rb") as f:
                    self.normaliser.stats = pickle.load(f)
            with open(self._cache_path(), "rb") as f:
                return pickle.load(f)
        return self._build_and_cache()

    # ── Download + extract ─────────────────────────────────────────────────────

    def _download_and_extract(self) -> tuple[str, str]:
        """Download gdb9.tar.gz and extract. Returns (sdf_path, csv_path)."""
        raw_dir  = os.path.join(self.root, "qm9_raw")
        os.makedirs(raw_dir, exist_ok=True)

        sdf_path = os.path.join(raw_dir, SDF_FILENAME)
        csv_path = os.path.join(raw_dir, CSV_FILENAME)

        if not os.path.exists(sdf_path) or not os.path.exists(csv_path):
            tar_path = os.path.join(raw_dir, "gdb9.tar.gz")
            if not os.path.exists(tar_path):
                print(f"Downloading QM9 (~80 MB) from {QM9_URL} ...")
                urllib.request.urlretrieve(QM9_URL, tar_path)
                print("Download complete.")
            print("Extracting archive...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(raw_dir)
            print("Extraction complete.")

        return sdf_path, csv_path

    # ── CSV reader ─────────────────────────────────────────────────────────────

    def _read_homo_lumo(self, csv_path: str) -> list[float]:
        """
        Return one HOMO-LUMO gap value per molecule row in the CSV.
        Index aligns 1-to-1 with SDF entry index (both skip header).
        """
        values: list[float] = []
        with open(csv_path, newline="") as fh:
            reader = csv.reader(fh)
            next(reader)  # skip header row
            for row in reader:
                try:
                    values.append(float(row[HOMO_LUMO_COL]))
                except (IndexError, ValueError):
                    values.append(float("nan"))
        return values

    # ── Main build logic ───────────────────────────────────────────────────────

    def _build_and_cache(self) -> list:
        """Download QM9, process molecules, split, normalise, cache all splits."""
        print(f"\nBuilding MolGen dataset (split={self.split}) ...")

        # Step 1 — acquire raw files
        sdf_path, csv_path = self._download_and_extract()

        # Step 2 — read HOMO-LUMO gaps (index matches SDF entry index)
        homo_lumo_vals = self._read_homo_lumo(csv_path)

        # Step 3 — parse SDF and build graphs, skipping malformed entries
        print("Parsing SDF and building molecular graphs...")
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        total = min(len(suppl), self.max_mols) if self.max_mols else len(suppl)

        all_data:  list = []
        all_props: list = []

        for i in tqdm(range(total), desc="Molecules"):
            mol = suppl[i]
            if mol is None:
                continue  # malformed SDF block — skip

            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue  # un-sanitisable molecule — skip

            smiles = Chem.MolToSmiles(mol, canonical=True)
            if not smiles:
                continue

            rdkit_props = compute_properties(smiles)
            if rdkit_props is None:
                continue

            # HOMO-LUMO gap from CSV row i (same index as SDF entry i)
            if i >= len(homo_lumo_vals):
                continue
            homo_lumo = homo_lumo_vals[i]
            if not math.isfinite(homo_lumo):
                continue
            rdkit_props["homo_lumo"] = homo_lumo

            graph = smiles_to_graph(smiles)
            if graph is None:
                continue

            all_data.append(graph)
            all_props.append(rdkit_props)

        print(f"Successfully built {len(all_data)} graphs.")

        # Step 4 — split indices 80 / 10 / 10
        n = len(all_data)
        indices = list(range(n))

        train_idx, temp_idx = train_test_split(
            indices,
            test_size=self.val_ratio + self.test_ratio,
            random_state=self.seed,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
            random_state=self.seed,
        )

        # Step 5 — fit normaliser on training split ONLY
        train_props = [all_props[i] for i in train_idx]
        self.normaliser.update_stats_from_dataset(train_props)

        # Step 6 — attach z-score normalised labels to every graph
        for graph, props in zip(all_data, all_props):
            normed = self.normaliser.normalise(props)
            graph.y = torch.tensor([[
                normed.get("logp",      0.0),
                normed.get("qed",       0.0),
                normed.get("sa_score",  0.0),
                normed.get("homo_lumo", 0.0),
            ]], dtype=torch.float)

        # Step 7 — cache all three splits + normaliser stats
        split_map = {
            "train": [all_data[i] for i in train_idx],
            "val":   [all_data[i] for i in val_idx],
            "test":  [all_data[i] for i in test_idx],
        }
        for name, data in split_map.items():
            path = os.path.join(self.root, f"molgen_{name}{self._suffix()}.pkl")
            with open(path, "wb") as f:
                pickle.dump(data, f)
            print(f"Cached {len(data):,} molecules → {path}")

        with open(self._normaliser_cache_path(), "wb") as f:
            pickle.dump(self.normaliser.stats, f)
        print(f"Cached normaliser stats → {self._normaliser_cache_path()}")

        return split_map[self.split]

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        return self.data_list[idx]


# ── Convenience function ───────────────────────────────────────────────────────

def get_dataloaders(
    root: str = "data",
    batch_size: int = 64,
    max_mols: Optional[int] = None,
    seed: int = 42,
) -> tuple:
    """
    Build train / val / test DataLoaders sharing one PropertyNormaliser.
    Normaliser stats are fitted on the training set only.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    shared_normaliser = PropertyNormaliser()

    train_ds = MolGenDataset(root=root, split="train",
                             normaliser=shared_normaliser,
                             max_mols=max_mols, seed=seed)
    val_ds   = MolGenDataset(root=root, split="val",
                             normaliser=shared_normaliser,
                             max_mols=max_mols, seed=seed)
    test_ds  = MolGenDataset(root=root, split="test",
                             normaliser=shared_normaliser,
                             max_mols=max_mols, seed=seed)

    train_loader = PyGDataLoader(train_ds, batch_size=batch_size,
                                 shuffle=True,  num_workers=0)
    val_loader   = PyGDataLoader(val_ds,   batch_size=batch_size,
                                 shuffle=False, num_workers=0)
    test_loader  = PyGDataLoader(test_ds,  batch_size=batch_size,
                                 shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
