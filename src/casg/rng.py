"""Deterministic, worker-independent RNG (protocol v3 section 21 `rng`).

Key = (base_seed, fold, epoch, sample_uid, view_id). The same key must
produce the same transform regardless of num_workers, shuffle order, or
resume point — that is hard gate "Simulator determinism" / "Multiworker
data loading" in section 22.

Implementation note: we hash the key with blake2b rather than combining
integers arithmetically, because sample_uid is a string and arithmetic
mixing of large ints collides in practice.
"""
from __future__ import annotations
import hashlib
import torch


def sample_seed(base_seed: int, fold: str, epoch: int, sample_uid: str,
                view_id: int) -> int:
    key = f"{int(base_seed)}|{fold}|{int(epoch)}|{sample_uid}|{int(view_id)}"
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & 0x7FFFFFFF


def sample_rng(base_seed: int, fold: str, epoch: int, sample_uid: str,
               view_id: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(sample_seed(base_seed, fold, epoch, sample_uid, view_id))
    return g
