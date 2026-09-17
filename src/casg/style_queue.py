"""CASG Module D: per-class FIFO style queue and far-style donor mining.

Every stored tensor is detached and lives on CPU (statistics in float16).
Donor rule (protocol section 8): same pathology class, different physical
patient, never outer-target data (the queue only ever sees training batches,
which the split construction already restricts). Selection: random sample
from the top style-distance quartile when ``hard`` else uniform.
"""
from __future__ import annotations
from collections import deque
from typing import Optional
import torch


class StyleQueue:
    def __init__(self, n_classes: int = 4, capacity: int = 256):
        self.q = {c: deque(maxlen=int(capacity)) for c in range(n_classes)}

    def __len__(self):
        return sum(len(d) for d in self.q.values())

    def enqueue(self, class_id: int, patient: int, style_code: torch.Tensor,
                mu_fd: torch.Tensor, log_sigma_fd: torch.Tensor):
        self.q[int(class_id)].append({
            "pid": int(patient),
            "s": style_code.detach().float().cpu(),
            "mu": mu_fd.detach().half().cpu(),
            "ls": log_sigma_fd.detach().half().cpu()})

    def sample(self, class_id: int, style_code: torch.Tensor, patient: int,
               hard: bool = False) -> Optional[dict]:
        cand = [it for it in self.q[int(class_id)] if it["pid"] != int(patient)]
        if not cand:
            return None
        s = style_code.detach().float().cpu()
        d = torch.tensor([1.0 - float(torch.dot(s, it["s"])) for it in cand])
        if hard and len(cand) >= 8:
            k = max(1, len(cand) // 4)
            top = torch.topk(d, k).indices
            pick = int(top[torch.randint(0, k, (1,)).item()])
        else:
            pick = int(torch.randint(0, len(cand), (1,)).item())
        it = cand[pick]
        return {"mu": it["mu"].float(), "ls": it["ls"].float(),
                "dist": float(d[pick])}
