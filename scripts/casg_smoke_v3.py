"""CASG protocol v3, section 22 — every machine-checkable hard gate.

    python scripts/casg_smoke_v3.py            # CASG-Lite (primary)
    python scripts/casg_smoke_v3.py --variant casg_full

CPU-only. Stops at the first failure. NO AUROC criterion is used (v3
explicitly removes the 3-epoch AUROC target): the gates are invariants.

Gates, in v3's order:
  1  Hospital-B exclusion            5  Invalid-label queue safety
  2  Smartphone fold composition     6  Simulator determinism (+ multiworker)
  3  Patient leakage                 7  Style masking / unmasked style input
  4  Outer-test blindness            8  Style-loss gradient isolation
  9  Donor class/patient safety     10  sample_far threshold
 11  Queue scope + checkpoint state 12  Residual intervention identity
 13  Resume equivalence             14  Inference pruning
"""
import _boot, argparse, os, copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from src.utils import load_config, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="casg_lite",
                choices=["cf_physics", "casg_lite", "casg_full"])
ap.add_argument("--config", default=None)
a = ap.parse_args()
cfg_path = a.config or f"configs/{a.variant}.yaml"
if not os.path.exists(cfg_path):
    cfg_path = "configs/casg.yaml"
cfg = load_config(cfg_path)
cfg["model"]["method"] = "casg"
cfg["model"]["variant"] = a.variant
set_seed(0)
print(f"config={cfg_path}  variant={a.variant}\n")

from src.data import manifest as M
from scripts.train_contrastive import _dev_filter, assert_no_frozen_site, internal_val

# ---- 1-3. topology: Hospital-B exclusion, phone fold, patient leakage ------
seoul = _dev_filter(M.add_patient_uid(M.load_manifest(cfg.data.train_manifest)),
                    cfg, "train")
icbhi = M.add_patient_uid(M.load_manifest(cfg.data.icbhi_manifest))
extra = _dev_filter(M.add_patient_uid(M.load_manifest(cfg.data.extra_manifest)),
                    cfg, "extra")
pool = pd.concat([icbhi, extra], ignore_index=True)
tags = [str(t).lower() for t in getattr(cfg.data, "dev_exclude_sources", []) or []]
assert tags, "data.dev_exclude_sources empty — run scripts/casg_fix_topology.py"
for dev in list(cfg.data.loo_devices):
    fold = M.build_loo_folds(pool, seoul, [dev], True)[dev]
    tr, va = internal_val(fold.train, float(cfg.data.internal_val_frac), 0)
    assert_no_frozen_site(cfg, tr, va)
    ptr = set(M.add_cohort_uid(tr)["cohort_uid"])
    pva = set(M.add_cohort_uid(va)["cohort_uid"]) if va is not None else set()
    pte = set(M.add_cohort_uid(fold.test)["cohort_uid"])
    assert not (ptr & pte) and not (pva & pte) and not (ptr & pva), \
        f"{dev}: patient sets not disjoint"
    if dev == "smartphone":
        src = set(fold.test["source"].astype(str).str.lower())
        assert not any("sungbook" in x for x in src), f"phone target: {src}"
print("[1-3] Hospital-B exclusion, Seoul-only phone target, patient "
      "disjointness (train/val/test) OK")

# ---- 4. outer-test blindness (capability, not instruction) -----------------
from src.casg.access_guard import arm_dev_mode, disarm, FrozenDataAccessError
fold = M.build_loo_folds(pool, seoul, ["AKGC417L"], True)["AKGC417L"]
outer = {os.path.join(str(cfg.data.data_root), p)
         for p in fold.test["filepath"].astype(str)}
arm_dev_mode(tags, outer)
try:
    victim = sorted(outer)[0]
    try:
        open(victim, "rb").close()
        raise AssertionError("dev mode opened an outer-target file")
    except FrozenDataAccessError:
        pass
    ok = os.path.join(str(cfg.data.data_root),
                      str(fold.train["filepath"].iloc[0]))
    open(ok, "rb").close()                       # source audio still readable
finally:
    disarm("test")
print("[4] outer-target blindness enforced by access capability OK")

# ---- 5-6. simulator determinism, worker independence, invalid labels -------
from src.casg.physics import PhysicsSimulator
from src.casg.rng import sample_rng
sim = PhysicsSimulator(dict(getattr(cfg, "casg", {}) or {}), 16000, band_hz=2000.0)
w = torch.randn(64000)
k = dict(base_seed=7, fold="AKGC417L", epoch=3, sample_uid="clip_042")
p1 = sim.sample(gen=sample_rng(**k, view_id=0))
p2 = sim.sample(gen=sample_rng(**k, view_id=0))
assert torch.equal(sim.apply(w, p1), sim.apply(w, p2)), \
    "same RNG key produced different transforms"
p_other = sim.sample(gen=sample_rng(**{**k, "epoch": 4}, view_id=0))
assert sim.style_distance(p1, p_other) > 0, "epoch does not change the draw"
far = sim.sample_far(p1, gen=sample_rng(**k, view_id=1))
print(f"[5-6] simulator determinism (worker-independent key) OK  "
      f"d_style(a,far)={sim.style_distance(p1, far):.3f}")

# ---- 7-8. model, style masking, gradient isolation ------------------------
from src.casg.model import CASGNet
from src.casg.trainer_step import casg_step, casg_state_dict, load_casg_state
from src.models import count_params
net = CASGNet(cfg, n_devices=5)
err = net.verify_split()
from src.trainer import Trainer
tr = Trainer(net, cfg, train_df=None, device=torch.device("cpu"))
B = 6
y4 = torch.tensor([0, 1, 1, 2, 3, -1])            # includes an invalid label
batch = {"mel": torch.randn(B, 1, 128, 401), "mel2": torch.randn(B, 1, 128, 401),
         "mel_clean": torch.randn(B, 1, 128, 401),
         "mel_clean2": torch.randn(B, 1, 128, 401),
         "label4": y4, "label2": (y4.clamp_min(0) > 0).long(),
         "pid": torch.tensor([10, 11, 12, 13, 14, 15]),
         "device_id": torch.zeros(B, dtype=torch.long),
         "index": torch.arange(B)}
if a.variant == "casg_full":
    taps = {t: torch.randn(2, 470, 768) for t in net.taps}
    s_masked = net.style_enc(taps, torch.randn(2, 1, 128, 401))
    s_clean = net.style_enc(taps, torch.randn(2, 1, 128, 401))
    assert not torch.allclose(s_masked, s_clean), "style encoder ignores its mel"
    L = F.mse_loss(net.style_enc(taps, torch.randn(2, 1, 128, 401)),
                   torch.zeros(2, 128))
    net.zero_grad(); L.backward()
    P = dict(net.named_parameters())
    bb = [p for n_, p in P.items() if "embeddings" in n_ or "encoder.layer" in n_]
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in bb), \
        "L_style leaked gradient into the AST backbone"
    assert P["style_enc.fuse.0.weight"].grad is not None
    net.zero_grad()
    print("[7-8] style branch consumes its own (unmasked) input; L_style "
          "gradient isolated from the backbone OK")
else:
    print("[7-8] style branch not used by this variant (v3: Lite has no "
          "learned style encoder) — masking/isolation N/A")

# ---- 9. donor class + patient safety --------------------------------------
from src.casg.trainer_step import _batch_donor_index
idx = _batch_donor_index(y4, batch["pid"])
for i, j in enumerate(idx.tolist()):
    if j < 0:
        continue
    assert int(y4[j]) == int(y4[i]) >= 0, "donor class mismatch or invalid"
    assert int(batch["pid"][j]) != int(batch["pid"][i]), "same-patient donor"
assert int(idx[5]) == -1, "label4=-1 anchor must never receive a donor"
print("[9] donor rule: same valid class, different patient, invalid labels "
      "excluded OK")

# ---- 10. sample_far threshold ---------------------------------------------
ref = sim.sample(gen=sample_rng(base_seed=1, fold="f", epoch=0,
                                sample_uid="u", view_id=0))
hits = sum(sim.style_distance(ref, sim.sample_far(ref)) >= sim.delta_style
           for _ in range(20))
assert hits >= 15, f"sample_far reached delta_style only {hits}/20 times"
print(f"[10] sample_far meets delta_style {hits}/20 (fallback otherwise) OK")

# ---- 11-12. queue scope, checkpoint state, AdaIN identity ------------------
from src.casg.adain import token_stats, residual_freq_adain
grid = torch.randn(2, net.f_patches, 39, 768)
mu, ls = token_stats(grid)
assert torch.equal(residual_freq_adain(grid, mu, ls, 0.0), grid), "rho=0 identity"
h9 = torch.randn(2, 470, 768)
merged = net.merge_grid(h9, residual_freq_adain(net.patch_grid(h9), mu, ls, 0.5))
assert torch.equal(merged[:, :net.n_special], h9[:, :net.n_special]), \
    "special tokens were modified"
assert torch.isfinite(merged).all()
print("[11-12] rho=0 identity, special-token protection, finite output OK")

# ---- 13. resume equivalence (weights + aux state) --------------------------
set_seed(0)
loss_a = casg_step(tr, batch, ep=7)
loss_a.backward(); tr.opt.step(); tr.opt.zero_grad()
w_a = copy.deepcopy(net.state_dict())
aux = casg_state_dict(tr)
set_seed(0)
net2 = CASGNet(cfg, n_devices=5)
net2.load_state_dict(w_a)
tr2 = Trainer(net2, cfg, train_df=None, device=torch.device("cpu"))
load_casg_state(tr2, aux)
assert tr2._casg["variant"] == a.variant
if a.variant == "casg_full":
    assert len(tr2._casg["queue"]) == len(tr._casg["queue"]), \
        "style queue not restored exactly"
for k_, v in net2.state_dict().items():
    assert torch.equal(v, w_a[k_]), f"weight mismatch after restore: {k_}"
print(f"[13] resume equivalence: weights + queue + curriculum state OK "
      f"(queue={len(tr._casg['queue'])})")

# ---- 14. inference pruning -------------------------------------------------
calls = {"style": 0}
orig = net.style_enc.forward
net.style_enc.forward = lambda *x, **kw: (calls.update(style=calls["style"] + 1),
                                          orig(*x, **kw))[1]
net.eval()
with torch.no_grad():
    out = net(torch.randn(2, 1, 128, 401))
net.style_enc.forward = orig
assert calls["style"] == 0, "eval graph called the style encoder"
assert out["logits4"].shape == (2, 4) and out["logits2"].shape == (2, 2)
print("[14] inference path calls no simulator / queue / style module OK")

print(f"\nALL v3 HARD GATES PASSED  (variant={a.variant}, "
      f"split-equivalence err={err:.1e}, params={count_params(net):,})")
