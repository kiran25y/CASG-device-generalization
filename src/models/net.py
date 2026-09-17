"""DeviceAgnosticNet: shared backbone + classifier heads + projection head
(contrastive) + optional device head (DANN). One model serves every method
(erm / dann / mixstyle / dmixstyle / coral / contrastive) so the comparison is
apples-to-apples on identical capacity.

MixStyle is applied here, on the log-mel input, so it is backbone-agnostic —
the previous arrangement wired it only into the hand-rolled CNN and silently
did nothing for the timm backbone used in every run.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .backbones import build_backbone
from .mixstyle import InputMixStyle


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


class DeviceAgnosticNet(nn.Module):
    def __init__(self, cfg, n_devices: int = 5):
        super().__init__()
        self.backbone = build_backbone(cfg)
        fd = self.backbone.feat_dim

        # MixStyle is active only for the methods that ask for it, so that the
        # 'mixstyle' row is a real baseline and 'erm' is a real control.
        method = str(getattr(cfg.model, "method", "contrastive")).lower()
        want = bool(getattr(cfg.model, "mixstyle", False)) or method in ("mixstyle", "dmixstyle")
        mode = "directional" if method == "dmixstyle" else "random"
        self.mixstyle = (InputMixStyle(p=float(getattr(cfg.model, "mixstyle_p", 0.5)),
                                       alpha=float(getattr(cfg.model, "mixstyle_alpha", 0.5)),
                                       mode=mode)
                         if want else None)

        self.h4 = nn.Linear(fd, int(cfg.model.n_classes4))
        self.h2 = nn.Linear(fd, int(cfg.model.n_classes2))
        self.proj = nn.Sequential(nn.Linear(fd, fd), nn.ReLU(True),
                                  nn.Linear(fd, int(cfg.model.proj_dim)))
        self.device_head = nn.Sequential(nn.Linear(fd, 128), nn.ReLU(True),
                                         nn.Dropout(0.3), nn.Linear(128, max(2, n_devices)))

    def features(self, mel, domain=None):
        if self.mixstyle is not None:
            mel = self.mixstyle(mel, domain)
        return self.backbone(mel)

    def forward(self, mel, domain=None):
        f = self.features(mel, domain)
        return {"logits4": self.h4(f), "logits2": self.h2(f), "feat": f}

    def project(self, mel):
        return F.normalize(self.proj(self.features(mel)), dim=-1)

    def device_logits(self, feat, lambd):
        return self.device_head(grad_reverse(feat, lambd))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
