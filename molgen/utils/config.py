"""
config.py

Config loading and validation using Pydantic.
Reads JSON config files and returns validated dicts.
"""

import json
from pathlib import Path


def load_encoder_config(path: str = "configs/encoder_config.json") -> dict:
    """
    Load and return the encoder config as a plain dict.
    Raises FileNotFoundError if the config doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Encoder config not found at: {path}")
    with open(p) as f:
        return json.load(f)


def load_diffusion_config(path: str = "configs/diffusion_config.json") -> dict:
    """Load and return the diffusion config as a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Diffusion config not found at: {path}")
    with open(p) as f:
        return json.load(f)