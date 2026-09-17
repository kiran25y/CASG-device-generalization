"""Capability-based blinding (protocol v3 sections 0, 4.6, 22).

v3 is explicit that blinding must be enforced "by access capability, not by
instructions to ignore output". This module installs a process-wide guard:
in development mode, any attempt to OPEN a Hospital-B / outer-target audio
file raises. Filtering a dataframe after loading is not sufficient, because
a cache, a style queue, or a stray script could still reach the bytes.

Usage (already wired by scripts/install_casg_v3.py):

    from src.casg.access_guard import arm_dev_mode
    arm_dev_mode(cfg, outer_device=held_out)   # dev entry point
    disarm()                                   # outer/external evaluators

The guard patches builtins.open, os.open and soundfile/torchaudio's file
readers at the path level: a path is forbidden when it matches any frozen
source tag or belongs to the current outer target's file set.
"""
from __future__ import annotations
import builtins
import os
from typing import Iterable, Optional, Set

_STATE = {"armed": False, "tags": (), "outer_paths": frozenset(),
          "orig_open": None, "orig_osopen": None, "mode": "dev"}


class FrozenDataAccessError(RuntimeError):
    """Raised when development code touches frozen or outer-target audio."""


def _forbidden(path: str) -> Optional[str]:
    if not _STATE["armed"]:
        return None
    p = str(path)
    low = p.lower().replace("\\", "/")
    for tag in _STATE["tags"]:
        if tag and tag in low:
            return f"frozen-site tag '{tag}'"
    if p in _STATE["outer_paths"] or os.path.abspath(p) in _STATE["outer_paths"]:
        return "outer LODO target"
    return None


def _guarded_open(file, *a, **kw):
    why = _forbidden(file) if isinstance(file, (str, bytes, os.PathLike)) else None
    if why:
        raise FrozenDataAccessError(
            f"DEVELOPMENT MODE may not open {file!r} ({why}). Protocol v3 "
            f"sections 4.6/22: blinding is enforced by access capability. "
            f"Use the separate outer/external evaluator after the freeze.")
    return _STATE["orig_open"](file, *a, **kw)


def _guarded_osopen(path, *a, **kw):
    why = _forbidden(path) if isinstance(path, (str, bytes, os.PathLike)) else None
    if why:
        raise FrozenDataAccessError(
            f"DEVELOPMENT MODE may not open {path!r} ({why}).")
    return _STATE["orig_osopen"](path, *a, **kw)


def arm_dev_mode(frozen_tags: Iterable[str],
                 outer_paths: Optional[Set[str]] = None) -> None:
    """Forbid every frozen-site path and (optionally) the outer target set."""
    tags = tuple(str(t).lower() for t in frozen_tags if str(t).strip())
    paths = set()
    for p in (outer_paths or ()):
        paths.add(str(p))
        paths.add(os.path.abspath(str(p)))
    if _STATE["orig_open"] is None:
        _STATE["orig_open"] = builtins.open
        _STATE["orig_osopen"] = os.open
    _STATE.update(armed=True, tags=tags, outer_paths=frozenset(paths),
                  mode="dev")
    builtins.open = _guarded_open
    os.open = _guarded_osopen
    print(f"[access-guard] DEV MODE armed: {len(tags)} frozen tag(s), "
          f"{len(paths) // 2} outer-target path(s) unreadable", flush=True)


def disarm(mode: str = "outer") -> None:
    if _STATE["orig_open"] is not None:
        builtins.open = _STATE["orig_open"]
        os.open = _STATE["orig_osopen"]
    _STATE.update(armed=False, mode=str(mode))
    print(f"[access-guard] disarmed (mode={mode})", flush=True)


def is_armed() -> bool:
    return bool(_STATE["armed"])


def require_freeze_token(path: str = "freeze.json") -> dict:
    """Outer/external evaluators refuse to run before the configuration is
    locked (protocol v3 sections 4.6, 24.4)."""
    import json
    if not os.path.exists(path):
        raise SystemExit(
            f"outer/external evaluation requires a freeze artifact ({path}). "
            f"Create it with scripts/casg_freeze.py AFTER source-only "
            f"development is complete. Protocol v3 section 24.4.")
    return json.load(open(path, encoding="utf-8"))
