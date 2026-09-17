"""CASG Module E: frequency-aware residual token style transfer.

Operates on the block-9 patch-token grid [B, F, T, H] (frequency-major token
order, special tokens excluded by the caller). Statistics are conditioned on
the frequency patch and computed over time (protocol section 9).
"""
from __future__ import annotations
import torch

_EPS = 1e-5


def token_stats(grid: torch.Tensor):
    """grid [B, F, T, H] -> (mu [B,F,H], log_sigma [B,F,H]) over time."""
    mu = grid.mean(dim=2)
    sigma = grid.std(dim=2) + _EPS
    return mu, torch.log(sigma)


def residual_freq_adain(grid: torch.Tensor, mu_d: torch.Tensor,
                        log_sigma_d: torch.Tensor, rho: float) -> torch.Tensor:
    """P_swap = P + rho * (AdaIN(P; donor stats) - P).  rho=0 -> identity."""
    if rho <= 0.0:
        return grid
    mu_i = grid.mean(dim=2, keepdim=True)
    sd_i = grid.std(dim=2, keepdim=True) + _EPS
    sd_d = torch.exp(log_sigma_d).unsqueeze(2)
    ad = (grid - mu_i) / sd_i * sd_d + mu_d.unsqueeze(2)
    return grid + float(rho) * (ad - grid)


def interpolate_stats(mu_i, log_sigma_i, mu_j, log_sigma_j, lam: float):
    """Virtual style: linear in mean, linear in log-sigma (protocol 9.1)."""
    return ((1 - lam) * mu_i + lam * mu_j,
            (1 - lam) * log_sigma_i + lam * log_sigma_j)
