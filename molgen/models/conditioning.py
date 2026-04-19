"""
conditioning.py

Property conditioning MLP for the graph diffusion model.

Takes target property values [logp, qed, sa_score, homo_lumo] and maps
them to a conditioning embedding vector that gets concatenated to node
features at every denoising step.

Also handles classifier-free guidance by supporting a learnable
null embedding — used when conditioning is dropped during training.
"""

import torch
import torch.nn as nn
from torch import Tensor


class PropertyConditioner(nn.Module):
    """
    Maps target property values to a conditioning embedding.

    During training:
        - 80% of steps: c = MLP(target_properties)
        - 20% of steps: c = null_embedding  (enables CFG at inference)

    At inference:
        - Run denoiser twice: once with c, once with null
        - Interpolate with guidance scale w

    Args:
        num_properties:   Number of input properties (4)
        conditioning_dim: Output embedding dimension (64)
        null_prob:        Probability of dropping conditioning during training (0.2)
    """

    def __init__(
        self,
        num_properties:   int = 4,
        conditioning_dim: int = 64,
        null_prob:        float = 0.2,
    ) -> None:
        super().__init__()
        self.conditioning_dim = conditioning_dim
        self.null_prob        = null_prob

        # MLP: 4 property values → 64-dim embedding
        self.mlp = nn.Sequential(
            nn.Linear(num_properties, conditioning_dim),
            nn.SiLU(),                                    # SiLU works better than ReLU for conditioning
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.SiLU(),
        )

        # Learned null embedding — used when conditioning is dropped
        # Initialised to zeros, learned during training
        self.null_embedding = nn.Parameter(
            torch.zeros(conditioning_dim)
        )

    def forward(
        self,
        properties: Tensor,
        force_null: bool = False,
    ) -> Tensor:
        """
        Compute conditioning embedding for a batch of property targets.

        Args:
            properties: Target property values   [B, 4]  (normalised)
            force_null: If True, always return null embedding (for CFG inference)

        Returns:
            Conditioning embedding               [B, conditioning_dim]
        """
        B = properties.shape[0]

        if force_null:
            return self.null_embedding.unsqueeze(0).expand(B, -1)

        # Compute real conditioning
        c = self.mlp(properties)   # [B, conditioning_dim]

        if self.training:
            # Randomly drop conditioning for classifier-free guidance training
            # Each item in the batch is independently dropped
            drop_mask = (torch.rand(B, device=properties.device) < self.null_prob)
            null = self.null_embedding.unsqueeze(0).expand(B, -1)
            c = torch.where(drop_mask.unsqueeze(-1), null, c)

        return c