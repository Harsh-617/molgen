"""
noise_schedule.py

Cosine noise schedule for the graph diffusion model.

Precomputes all the constants needed for the forward process:
    alphas_bar:     signal retention at each timestep  [T+1]
    alphas:         per-step noise multiplier          [T]
    betas:          per-step noise added               [T]
    sqrt_ab:        sqrt(alphas_bar)                   [T+1]
    sqrt_one_minus_ab: sqrt(1 - alphas_bar)            [T+1]
"""

import torch
import math


class CosineNoiseSchedule:
    """
    Precomputed cosine noise schedule following Nichol & Dhariwal 2021.

    The cosine schedule corrupts data more gently than linear — it spends
    more timesteps near the clean signal where the model can learn structure,
    and fewer steps in the fully-corrupted regime where everything is noise.

    All tensors are stored on CPU and moved to device on demand.

    Args:
        T:   Total diffusion timesteps (default 1000)
        s:   Small offset to prevent singularity at t=0 (default 0.008)
    """

    def __init__(self, T: int = 1000, s: float = 0.008) -> None:
        self.T = T

        # ── Compute alphas_bar ─────────────────────────────────────────────
        # f(t) = cos²( (t/T + s) / (1 + s) * π/2 )
        # alphas_bar[t] = f(t) / f(0)   so alphas_bar[0] = 1.0
        t = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((t / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alphas_bar = f / f[0]

        # Clamp to avoid numerical issues at the boundaries
        alphas_bar = alphas_bar.clamp(1e-5, 1.0 - 1e-5).float()

        # ── Derived quantities ─────────────────────────────────────────────
        # alphas[t] = alphas_bar[t] / alphas_bar[t-1]  for t = 1..T
        alphas = alphas_bar[1:] / alphas_bar[:-1]       # [T]
        betas  = 1.0 - alphas                            # [T]

        # Store everything — indexed by timestep t ∈ [0, T]
        self.alphas_bar          = alphas_bar            # [T+1]
        self.alphas              = alphas                # [T]
        self.betas               = betas                 # [T]
        self.sqrt_ab             = alphas_bar.sqrt()     # [T+1]
        self.sqrt_one_minus_ab   = (1.0 - alphas_bar).sqrt()  # [T+1]

        # For the reverse process posterior (DDPM sampling)
        # posterior_variance[t] = betas[t] * (1 - alphas_bar[t-1]) / (1 - alphas_bar[t])
        ab      = alphas_bar[1:]    # alphas_bar[t]      [T]
        ab_prev = alphas_bar[:-1]   # alphas_bar[t-1]    [T]
        self.posterior_variance = (betas * (1.0 - ab_prev) / (1.0 - ab)).clamp(1e-20)

    def to(self, device: torch.device) -> "CosineNoiseSchedule":
        """Move all tensors to device. Returns self for chaining."""
        self.alphas_bar        = self.alphas_bar.to(device)
        self.alphas            = self.alphas.to(device)
        self.betas             = self.betas.to(device)
        self.sqrt_ab           = self.sqrt_ab.to(device)
        self.sqrt_one_minus_ab = self.sqrt_one_minus_ab.to(device)
        self.posterior_variance= self.posterior_variance.to(device)
        return self

    # ── Forward process ────────────────────────────────────────────────────

    def q_sample(
        self,
        x0:  torch.Tensor,
        t:   torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from q(x_t | x_0) — the forward noising process.

        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε

        Args:
            x0:  Clean node features           [N, feat_dim]
            t:   Timestep per node             [N] — same t for all nodes in a graph
            eps: Optional pre-sampled noise    [N, feat_dim]
                 If None, samples fresh Gaussian noise.

        Returns:
            (x_t, eps) — noisy features and the noise that was added.
            We return eps so the denoiser can be trained to predict it.
        """
        if eps is None:
            eps = torch.randn_like(x0)

        # Index into schedule using per-node timesteps
        sqrt_ab       = self.sqrt_ab[t].unsqueeze(-1)        # [N, 1]
        sqrt_one_minus= self.sqrt_one_minus_ab[t].unsqueeze(-1)  # [N, 1]

        x_t = sqrt_ab * x0 + sqrt_one_minus * eps
        return x_t, eps

    def corrupt_adjacency(
        self,
        A0:  torch.Tensor,
        t:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Corrupt the adjacency matrix using discrete bit-flipping.

        Each edge independently flips (0→1 or 1→0) with probability beta_t.
        At t=0: A unchanged. At t=T: A is nearly random Bernoulli(0.5).

        Args:
            A0: Clean adjacency matrix   [N, N] — binary (0 or 1)
            t:  Scalar timestep          int or 0-dim tensor

        Returns:
            A_t: Corrupted adjacency     [N, N] — binary
        """
        t_idx = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        # Cumulative flip probability: probability each bit has been flipped
        # p_flip(t) = 0.5 * (1 - alphas_bar[t])  — derived from discrete diffusion
        p_flip = 0.5 * (1.0 - self.alphas_bar[t_idx].item())

        flip_mask = (torch.rand_like(A0.float()) < p_flip)
        A_t = (A0.float() + flip_mask.float()) % 2   # XOR via mod-2 addition
        return A_t.long()

    # ── Reverse process ────────────────────────────────────────────────────

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t:   torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover predicted x_0 from predicted noise eps.

        x_0_hat = (x_t - sqrt(1-α̅_t) * eps) / sqrt(α̅_t)

        Args:
            x_t: Noisy features at timestep t    [N, feat_dim]
            t:   Timestep per node               [N]
            eps: Predicted noise                 [N, feat_dim]

        Returns:
            x_0_hat: Predicted clean features    [N, feat_dim]
        """
        sqrt_ab        = self.sqrt_ab[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_ab[t].unsqueeze(-1)
        return (x_t - sqrt_one_minus * eps) / sqrt_ab.clamp(min=1e-8)

    def ddpm_step(
        self,
        x_t: torch.Tensor,
        t:   int,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """
        One reverse diffusion step: sample x_{t-1} from x_t.

        Uses the DDPM posterior:
            x_{t-1} = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-α̅_t) * eps_pred)
                      + sqrt(posterior_variance_t) * z,  z ~ N(0,I)

        Args:
            x_t: Noisy features at step t     [N, feat_dim]
            t:   Current timestep (int)
            eps: Predicted noise from model   [N, feat_dim]

        Returns:
            x_{t-1}: Slightly denoised features  [N, feat_dim]
        """
        beta_t    = self.betas[t - 1]
        alpha_t   = self.alphas[t - 1]
        ab_t      = self.alphas_bar[t]

        # Mean of the posterior
        coef = beta_t / (1.0 - ab_t).sqrt()
        mean = (1.0 / alpha_t.sqrt()) * (x_t - coef * eps)

        if t == 1:
            return mean   # no noise added at the final step

        var = self.posterior_variance[t - 1]
        z   = torch.randn_like(x_t)
        return mean + var.sqrt() * z