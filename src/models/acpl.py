"""ACPL-v1: Acquisition-Conditioned Pathology Learning.

Implements exactly the three losses of the experiment report and nothing more:

    L = L_cls + lambda_acq * L_acq + lambda_CF * L_CF

  L_cls   focal 4-class + binary pathology loss on the acquisition-conditioned
          features, averaged over both synthetic-device views.
  L_acq   MSE between the acquisition head's estimate theta_hat and the TRUE
          normalised transformation parameters theta emitted by
          DeviceSimulator.call_with_theta(). This is free supervision: theta is
          known exactly because we drew it.
  L_CF    counterfactual consistency. The report writes
          KL[p(y|x) || p(y|A(x))]; implemented with two deliberate deviations,
          both stated in the manuscript when this ships:
            1. STOP-GRADIENT on the teacher side. Without it, the cheapest
               minimum is degrading the clean prediction to match the corrupted
               one; consistency training standardly detaches the teacher.
            2. Consistency is enforced BETWEEN THE TWO SYNTHETIC-DEVICE VIEWS
               (symmetrised), not clean-vs-augmented, because a "clean" clip is
               itself one acquisition state, and the dataset already produces
               two views per clip for SupCon at no extra forward cost. Set
               model.acpl_clean_anchor: true to add a third, un-augmented view
               and anchor on it instead (one extra forward pass per step).

Architecture (report Table 7, stages C-F): the shared backbone yields h; a
pathology projection gives z_p, an acquisition encoder gives q; a FiLM layer
conditions z_p on q before the shared classifier heads. At inference q is
computed from the input itself, so no target metadata is ever needed.

The classifier conditioning on q is NOT a leak under strict DG: q is a function
of the audio, and its role is to let the classifier normalise acquisition
effects out, while L_CF forbids it from using them as a shortcut.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import DeviceAgnosticNet


class ACPLNet(DeviceAgnosticNet):
    def __init__(self, cfg, n_devices: int, theta_dim: int):
        super().__init__(cfg, n_devices)
        fd = self.backbone.feat_dim
        dq = int(getattr(cfg.model, "acq_dim", 64))
        self.theta_dim = int(theta_dim)

        # pathology projection z_p (probed in E11) and acquisition encoder q
        self.path_proj = nn.Sequential(nn.Linear(fd, fd), nn.GELU(),
                                       nn.Linear(fd, fd))
        self.acq_enc = nn.Sequential(nn.Linear(fd, 256), nn.GELU(),
                                     nn.Linear(256, dq))
        self.theta_head = nn.Linear(dq, self.theta_dim)
        # FiLM conditioning of z_p on q; initialised to identity so training
        # starts exactly at the ERM solution surface
        self.film = nn.Linear(dq, 2 * fd)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    # ---- stages ------------------------------------------------------------
    def acq_state(self, feat):
        q = self.acq_enc(feat)
        return q, self.theta_head(q)

    def condition(self, feat, q):
        zp = self.path_proj(feat)
        g, b = self.film(q).chunk(2, dim=-1)
        return zp * (1.0 + g) + b, zp

    def acpl_forward(self, mel, domain=None):
        f = self.features(mel, domain)
        q, theta_hat = self.acq_state(f)
        zc, zp = self.condition(f, q)
        return {"logits4": self.h4(zc), "logits2": self.h2(zc),
                "feat": f, "z_p": zp, "q": q,
                "theta_hat": torch.sigmoid(theta_hat)}   # targets live in [0,1]

    # keep the generic eval path working: forward() = conditioned prediction
    def forward(self, mel, domain=None):
        out = self.acpl_forward(mel, domain)
        return {"logits4": out["logits4"], "logits2": out["logits2"],
                "feat": out["feat"]}


# ------------------------------------------------------------------ losses --
def acq_loss(theta_hat, theta):
    """Normalised-parameter regression. Both sides live in [0, 1] per
    dimension (see DeviceSimulator.call_with_theta), so plain MSE is balanced
    across dB / Hz / bits / SNR components by construction."""
    return F.mse_loss(theta_hat, theta)


def cf_consistency(logits_a, logits_b):
    """Symmetrised stop-gradient KL between the two acquisition views.

    KL[sg(p_a) || p_b] + KL[sg(p_b) || p_a], averaged. The detached side is
    the teacher; each view teaches the other, so neither can converge by
    degrading its own prediction."""
    pa = F.log_softmax(logits_a, dim=-1)
    pb = F.log_softmax(logits_b, dim=-1)
    kl_ab = F.kl_div(pb, pa.detach().exp(), reduction="batchmean")
    kl_ba = F.kl_div(pa, pb.detach().exp(), reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba)


def cf_consistency_anchored(logits_clean, logits_aug):
    """The report's literal form, teacher = the un-augmented view, detached:
    KL[sg(p(y|x)) || p(y|A(x))]. Used when acpl_clean_anchor is enabled."""
    p_clean = F.log_softmax(logits_clean, dim=-1).detach().exp()
    p_aug = F.log_softmax(logits_aug, dim=-1)
    return F.kl_div(p_aug, p_clean, reduction="batchmean")
