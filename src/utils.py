"""Config (attribute-accessible YAML) + reproducibility."""
from __future__ import annotations
import copy, os, random
from typing import Any, Dict
import numpy as np


class Config(dict):
    def __init__(self, data: Dict[str, Any] | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def to_dict(self):
        def un(v): return {k: un(x) for k, x in v.items()} if isinstance(v, Config) else copy.deepcopy(v)
        return {k: un(v) for k, v in self.items()}


def load_config(path: str) -> Config:
    import yaml
    with open(path, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
