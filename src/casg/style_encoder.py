"""CASG Module C: training-only style encoder (protocol section 7.2).

Consumes DETACHED AST token taps (blocks 3/6/9) plus raw log-mel statistics
and produces a 128-D L2-normalized continuous style code. The taps are
detached by the CALLER; this module never touches the backbone's gradients.
Supports a token-time window so the style-contrastive triplet can compare
temporal crops without extra forward passes.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleEncoder(nn.Module):
    def __init__(self, hidden: int = 768, mel_bins: int = 128,
                 f_patches: int = 12, n_special: int = 2,
                 taps=(3, 6, 9), style_dim: int = 128):
        super().__init__()
        self.f = int(f_patches)
        self.ns = int(n_special)
        self.taps = tuple(int(t) for t in taps)
        self.proj = nn.ModuleDict({str(t): nn.Linear(hidden, 64)
                                   for t in self.taps})
        # per tap: freq-profile mean+std over frequency patches -> 128
        tap_out = 128 * len(self.taps)
        self.mel_mlp = nn.Sequential(nn.Linear(2 * mel_bins, 128), nn.GELU(),
                                     nn.Linear(128, 128))
        self.fuse = nn.Sequential(nn.Linear(tap_out + 128, 256), nn.GELU(),
                                  nn.Linear(256, style_dim))

    def _tap_summary(self, tokens: torch.Tensor, t_slice) -> torch.Tensor:
        # tokens [B, ns + f*t, H] -> patch grid [B, f, t, H] (frequency-major)
        b, n, h = tokens.shape
        patch = tokens[:, self.ns:, :]
        t = patch.shape[1] // self.f
        grid = patch.reshape(b, self.f, t, h)
        if t_slice is not None:
            grid = grid[:, :, t_slice[0]:t_slice[1], :]
        return grid.mean(dim=2)                       # [B, f, H] time-pooled

    def forward(self, taps: Dict[int, torch.Tensor], mel: torch.Tensor,
                t_slice: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        parts = []
        n_tok = (taps[self.taps[0]].shape[1] - self.ns) // self.f
        for tnum in self.taps:
            g = self.proj[str(tnum)](self._tap_summary(taps[tnum], t_slice))
            parts.append(torch.cat([g.mean(dim=1), g.std(dim=1)], dim=-1))
        # raw log-mel frequency statistics over the matching time window
        m = mel.squeeze(1) if mel.dim() == 4 else mel  # [B, F, T]
        if t_slice is not None:
            # token-time -> mel-frame mapping: proportional slice
            tt = m.shape[-1]
            a = int(t_slice[0] / n_tok * tt)
            b = max(a + 1, int(min(t_slice[1], n_tok) / n_tok * tt))
            m = m[..., a:b]
        parts.append(self.mel_mlp(torch.cat([m.mean(dim=-1), m.std(dim=-1)],
                                            dim=-1)))
        return F.normalize(self.fuse(torch.cat(parts, dim=-1)), dim=-1)


def style_info_nce(q: torch.Tensor, p: torch.Tensor, n: torch.Tensor,
                   tau: float = 0.07) -> torch.Tensor:
    """One strong same-content/different-style negative per anchor, plus the
    other in-batch negatives (protocol section 7.1)."""
    pos = (q * p).sum(-1, keepdim=True) / tau                     # [B,1]
    neg_own = (q * n).sum(-1, keepdim=True) / tau                 # [B,1]
    neg_x = (q @ n.t()) / tau                                     # [B,B]
    logits = torch.cat([pos, neg_own, neg_x], dim=1)
    return F.cross_entropy(logits, torch.zeros(q.shape[0], dtype=torch.long,
                                               device=q.device))
