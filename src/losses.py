"""Losses: logit-adjusted focal/CE (sensitivity fix), device-augmented
supervised contrastive (device-label-free), and Deep CORAL.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def focal_loss(logits, target, gamma=2.0, adjust=None):
    if adjust is not None:                        # logit adjustment: += tau*log_prior
        logits = logits + adjust.to(logits.device)
    logp = F.log_softmax(logits, -1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    return (-((1 - pt) ** gamma) * logpt).mean()


def ce_loss(logits, target, adjust=None):
    if adjust is not None:
        logits = logits + adjust.to(logits.device)
    return F.cross_entropy(logits, target)


def pathology_loss(logits4, logits2, y4, y2, gamma=2.0, w4=0.7, w2=0.3,
                   adjust4=None, adjust2=None):
    mask = y4 >= 0                                # extra binary/disease data: y4 == -1
    l4 = focal_loss(logits4[mask], y4[mask], gamma, adjust4) if mask.any() \
        else logits4.sum() * 0.0
    l2 = ce_loss(logits2, y2, adjust2)
    return w4 * l4 + w2 * l2


def supcon_loss(feats, labels, tau=0.1):
    """Supervised contrastive (Khosla et al.). feats [N,D] L2-normalised,
    labels [N]. Positives = same-label samples, which includes the two
    device-augmented views of the same clip. Device-label-free."""
    device = feats.device
    n = feats.size(0)
    sim = (feats @ feats.t()) / tau
    sim = sim - sim.max(1, keepdim=True)[0].detach()
    self_mask = torch.eye(n, device=device, dtype=torch.bool)
    exp = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)
    pos_cnt = pos.sum(1).clamp_min(1)
    mlpp = (pos.float() * log_prob).sum(1) / pos_cnt
    valid = pos.sum(1) > 0
    return -(mlpp[valid].mean()) if valid.any() else feats.sum() * 0.0


def compute_logit_adjust(class_counts, tau=1.0):
    """tau * log(prior); added to logits during training, raw logits at test."""
    p = torch.as_tensor(class_counts, dtype=torch.float32)
    p = p / p.sum().clamp_min(1)
    return tau * torch.log(p.clamp_min(1e-8))


def _cov(x):
    xm = x - x.mean(0, keepdim=True)
    return (xm.t() @ xm) / (x.size(0) - 1 + 1e-6)


def coral_loss(source, target):
    """Deep CORAL feature-covariance alignment between two DOMAINS."""
    d = source.size(1)
    return (_cov(source) - _cov(target)).pow(2).sum() / (4 * d * d)


def coral_by_domain(feats, domain_ids, min_per_domain: int = 4):
    """Average pairwise Deep CORAL over the domains present in the batch.

    The previous implementation aligned ``f[:h]`` against ``f[h:2h]`` — two
    arbitrary halves of a shuffled batch, i.e. a distribution against itself.
    With batch_size 16 that was two 8-sample estimates of a 256x256 covariance
    (rank <= 7), so the penalty was pure estimator noise; measured against ERM
    on the paired cells it cost Se2 -0.040 and Se4 -0.039 for no principled
    reason, which unfairly penalised a published baseline.

    This version groups by actual device id. Domains with fewer than
    ``min_per_domain`` samples in the batch are skipped. Use the
    ``device_class`` sampler (and a batch large enough to hold several devices)
    so that most batches contain >= 2 usable domains.
    """
    uniq = torch.unique(domain_ids)
    groups = []
    for d in uniq:
        idx = (domain_ids == d).nonzero(as_tuple=False).view(-1)
        if idx.numel() >= min_per_domain:
            groups.append(feats[idx])
    if len(groups) < 2:
        return feats.sum() * 0.0
    total, n = feats.sum() * 0.0, 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            total = total + coral_loss(groups[i], groups[j])
            n += 1
    return total / max(1, n)
