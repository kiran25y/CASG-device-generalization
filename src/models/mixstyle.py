"""MixStyle on the log-mel input, with per-FREQUENCY statistics.

Why this file was rewritten
---------------------------
The previous version was a conventional feature-map MixStyle that
``build_backbone`` only ever wired into the CNN path. With ``backbone: timm``
(our default for every run to date) it was never called: `runs/comparison.csv`
shows every ``mixstyle`` row bit-identical to the corresponding ``erm`` row
across 5 devices x 3 seeds. The MixStyle baseline was ERM under another name.

Two things change here:

1. **Placement.** MixStyle is applied to the log-mel tensor immediately before
   the backbone, so it works for every backbone (timm CNN *and* AST) rather
   than only for the hand-rolled CNN.

2. **Statistics axis.** Device discrepancy in this corpus is overwhelmingly a
   frequency-axis phenomenon (spectral centroid 77 Hz for the Littmann CORE vs
   959 Hz for the iPhone on paired recordings). So statistics are per-frequency
   — for each mel bin, mean and std over time — not per-channel over the whole
   map.

Two modes
---------
``random``      : classic domain-generalisation MixStyle. Mixes each sample with
                  a random other sample in the batch. Needs no device labels and
                  no target data — usable in our LODO protocol.

``directional`` : source -> target adaptation. Only source samples are restyled,
                  using statistics drawn from target-domain samples; target
                  samples pass through untouched. This REQUIRES target-domain
                  data in the training batch, so it is only valid for the oracle
                  upper-bound row (train with the deployment device visible), not
                  for leave-one-device-out.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class InputMixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.5, mode: str = "random",
                 eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.eps = float(eps)
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)

    def extra_repr(self):
        return f"p={self.p}, alpha={self.alpha}, mode={self.mode}"

    @staticmethod
    def _stats(x, eps):
        """Per-frequency mean/std: reduce over time only. x is [B, C, F, T]."""
        mu = x.mean(dim=-1, keepdim=True)
        sig = (x.var(dim=-1, keepdim=True) + eps).sqrt()
        return mu, sig

    def forward(self, x, domain: torch.Tensor | None = None):
        """x: [B, C, F, T] log-mel.
        domain: optional [B] bool/long mask, True (or 1) == TARGET domain.
                Only consulted in 'directional' mode.
        """
        if not self.training or x.size(0) < 2:
            return x
        if torch.rand(1).item() > self.p:
            return x

        B = x.size(0)
        mu, sig = self._stats(x, self.eps)
        xn = (x - mu) / sig

        if self.mode == "directional":
            if domain is None:
                return x
            tgt = domain.bool().view(-1)
            src = ~tgt
            if tgt.sum() < 1 or src.sum() < 1:
                return x                                  # need both sides
            tgt_idx = torch.nonzero(tgt, as_tuple=False).view(-1)
            # every sample draws a random TARGET partner; targets are left alone
            pick = tgt_idx[torch.randint(len(tgt_idx), (B,), device=x.device)]
            lam = self.beta.sample((B, 1, 1, 1)).to(x.device).to(x.dtype)
            mu_hat = lam * mu + (1 - lam) * mu[pick]
            sig_hat = lam * sig + (1 - lam) * sig[pick]
            out = xn * sig_hat + mu_hat
            keep = tgt.view(B, 1, 1, 1)
            return torch.where(keep, x, out)              # targets pass through

        # ---- random (domain-generalisation) mode --------------------------
        perm = torch.randperm(B, device=x.device)
        lam = self.beta.sample((B, 1, 1, 1)).to(x.device).to(x.dtype)
        return xn * (lam * sig + (1 - lam) * sig[perm]) + (lam * mu + (1 - lam) * mu[perm])


# Backwards-compatible alias: older checkpoints / imports referenced `MixStyle`.
MixStyle = InputMixStyle
