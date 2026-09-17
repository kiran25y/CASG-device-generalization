"""v2 experiment gates. Every check runs the REAL loader / REAL step and
asserts on tensors. A gate that inspects source text is not a gate.

    python scripts/v2_gates.py                 # all gates, small model on CPU/GPU
    python scripts/v2_gates.py --n 40          # more loader samples

Gates
  G1  shared masks: mask positions identical across the two paired views
      for every sampled example (frequency rows AND time columns).
  G2  cohort donor exclusion: in casg_lite, no donor shares cohort_uid with
      its anchor, and pid == stable_hash(cohort_uid) in the batch.
  G3  identity control: with variant=casg_identity at an intervention
      epoch, the third view's patch grid is bit-identical to the anchor's,
      the branch IS executed (3 views), and its logits are NOT identical to
      the anchor's (independent dropout realisation).
  G4  intervention active: with variant=casg_lite at an intervention epoch,
      the third view differs from the anchor for anchors that have a donor
      and equals it for anchors that do not (explicit identity path).
  G5  schedule: before random_swap_start both intervention variants produce
      exactly two views; cf_physics never produces a third.
  G6  EMA: after one epoch past ema_start, the trainer holds an EMA state
      that differs from the raw weights and has ema_updates > 0.
  G7  loss weights: w_ab / w_as read from casg.lambda_cf_ab / lambda_cf_as
      and default to lambda_cf.
Exit code 1 on the first failure.
"""
import _boot, argparse, sys, hashlib
import numpy as np, pandas as pd, torch
from src.utils import load_config, set_seed
from src.data import manifest as M
from src.data.dataset import RSCDataset, make_loader
from src.trainer import Trainer, resolve_device
from scripts.train_contrastive import build_pool, _dev_filter

FAILS = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def masks_of(m):
    m = m.squeeze()
    rows = frozenset(i for i in range(m.shape[0]) if float(m[i].std()) < 1e-6)
    cols = frozenset(j for j in range(m.shape[1]) if float(m[:, j].std()) < 1e-6)
    return rows, cols


def stable_hash(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") >> 1


def cset(node, key, val):
    """Set a config key whether the node is attribute- or dict-style."""
    try:
        setattr(node, key, val)
    except Exception:
        node[key] = val
    got = node[key] if isinstance(node, dict) else getattr(node, key)
    assert got == val, f"could not set {key}"


def build_ds(cfg, n_rows=256):
    seoul, pool = build_pool(cfg); seoul = _dev_filter(seoul, cfg, "train")
    df = pd.concat([seoul, pool], ignore_index=True)
    df = M.add_cohort_uid(M.add_patient_uid(df))
    # a small, class-balanced, multi-cohort slice so donors exist in-batch
    parts = [g.head(max(8, n_rows // 8)) for _, g in df.groupby(["device", "label4"])]
    small = pd.concat(parts).sample(frac=1.0, random_state=0).head(n_rows).reset_index(drop=True)
    dmap = M.device_to_id(df)
    return small, dmap, cfg.data.data_root


def tiny_model(cfg, dmap):
    from src.casg.model import CASGNet
    return CASGNet(cfg, n_devices=max(2, len(dmap)))


def one_batch(small, cfg, dmap, root, bs=16):
    ds = RSCDataset(small, cfg, dmap, root, train=True)
    ds.casg_epoch = 0
    items = [ds[i] for i in range(bs)]
    out = {}
    for k in items[0]:
        v = items[0][k]
        if isinstance(v, torch.Tensor):
            out[k] = torch.stack([it[k] for it in items])
        else:
            out[k] = [it[k] for it in items]
    return ds, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/casg_lite.yaml")
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    set_seed(0)
    cfg = load_config(a.config)
    dev = resolve_device(cfg)
    small, dmap, root = build_ds(cfg)
    swap_start = int(cfg.casg.curriculum.random_swap_start)
    print(f"gates on {len(small)} clips, device={dev}, random_swap_start={swap_start}\n")

    # ---------------------------------------------------------------- G1
    ds = RSCDataset(small, cfg, dmap, root, train=True); ds.casg_epoch = 3
    same = 0; first = None
    for i in range(a.n):
        it = ds[i]
        ra, ca = masks_of(it["mel"]); rb, cb = masks_of(it["mel2"])
        ok = (ra == rb) and (ca == cb); same += ok
        if first is None and not ok:
            first = (sorted(ra)[:6], sorted(rb)[:6])
    gate("G1 shared SpecAugment mask positions across paired views",
         same == a.n, f"{same}/{a.n} identical" + (f"; first mismatch rows {first}" if first else ""))

    # ---------------------------------------------------------------- G2
    ds, b = one_batch(small, cfg, dmap, root, bs=32)
    coh = [str(small.iloc[int(ix)]["cohort_uid"]) for ix in b["index"]]
    pid_ok = all(int(b["pid"][k]) == stable_hash(coh[k]) for k in range(len(coh)))
    gate("G2a batch pid == stable_hash(cohort_uid)", pid_ok)
    from src.casg.trainer_step import _batch_donor_index
    idx = _batch_donor_index(b["label4"], b["pid"])
    viol = [(k, int(idx[k])) for k in range(len(coh)) if idx[k] >= 0 and coh[int(idx[k])] == coh[k]]
    n_don = int((idx >= 0).sum())
    gate("G2b no donor shares cohort_uid with its anchor", not viol and n_don > 0,
         f"{n_don}/{len(coh)} anchors with donor, {len(viol)} violations")

    # ---------------------------------------------------------- G3/G4/G5
    from src.casg import trainer_step as TS
    def run_step(variant, ep):
        c = load_config(a.config); cset(c.model, "variant", variant)
        m = tiny_model(c, dmap).to(dev); m.train()
        tr = Trainer(m, c, train_df=None, device=dev)
        if hasattr(tr, "_casg"): del tr._casg
        # instrument: capture the concatenated views by wrapping forward_late
        captured = {}
        orig_late = m.forward_late
        def late(x):
            captured["cat"] = x.detach().clone(); return orig_late(x)
        m.forward_late = late
        with torch.no_grad():
            loss = TS.casg_step(tr, b, ep)
        B = b["mel"].shape[0]
        nv = captured["cat"].shape[0] // B
        views = [captured["cat"][k * B:(k + 1) * B] for k in range(nv)]
        return tr, m, loss, views, B

    torch.manual_seed(0)
    tr_id, m_id, _, v_id, B = run_step("casg_identity", swap_start - 1)
    gate("G3a identity: three views at intervention epoch", len(v_id) == 3, f"views={len(v_id)}")
    if len(v_id) == 3:
        ga = m_id.patch_grid(v_id[0]); gs = m_id.patch_grid(v_id[2])
        gate("G3b identity: third-view patch grid bit-identical to anchor grid",
             torch.equal(ga, gs), f"max|diff|={float((ga - gs).abs().max()):.2e}")
        ca = v_id[0][:, :m_id.n_special]; cs = v_id[2][:, :m_id.n_special]
        gate("G3c identity: special tokens untouched", torch.equal(ca, cs))
        # independent dropout: forward the concatenated views again and check
        # logits of view a vs view s differ (dropout active in train mode)
        m_id.train()
        with torch.no_grad():
            pooled = m_id.forward_late(torch.cat([v_id[0], v_id[2]], 0))
            l4, l2, _ = m_id.heads_from_pooled(pooled)
        gate("G3d identity: branch has its own dropout realisation (logits differ)",
             not torch.allclose(l2[:B], l2[B:]), f"max|Δlogit|={float((l2[:B]-l2[B:]).abs().max()):.2e}")

    torch.manual_seed(0)
    tr_lt, m_lt, _, v_lt, B = run_step("casg_lite", swap_start - 1)
    gate("G4a lite: three views at intervention epoch", len(v_lt) == 3, f"views={len(v_lt)}")
    if len(v_lt) == 3:
        idx = TS._batch_donor_index(b["label4"], b["pid"])
        have = (idx >= 0)
        ga = m_lt.patch_grid(v_lt[0]); gs = m_lt.patch_grid(v_lt[2])
        d = (ga - gs).flatten(1).abs().max(dim=1).values.cpu()
        moved = d[have] > 1e-6; kept = d[~have] < 1e-7
        gate("G4b lite: donor-eligible anchors are transformed",
             bool(moved.all()) if have.any() else True,
             f"{int(moved.sum())}/{int(have.sum())} eligible anchors moved")
        gate("G4c lite: ineligible anchors take the explicit identity path",
             bool(kept.all()) if (~have).any() else True,
             f"{int(kept.sum())}/{int((~have).sum())} ineligible anchors unchanged")

    torch.manual_seed(0)
    _, _, _, v_pre, _ = run_step("casg_lite", swap_start - 2)
    _, _, _, v_pre_id, _ = run_step("casg_identity", swap_start - 2)
    _, _, _, v_cf, _ = run_step("cf_physics", swap_start + 5)
    gate("G5 schedule: 2 views before swap start (lite, identity); cf_physics never 3",
         len(v_pre) == 2 and len(v_pre_id) == 2 and len(v_cf) == 2,
         f"lite={len(v_pre)} identity={len(v_pre_id)} cf={len(v_cf)}")

    # ---------------------------------------------------------------- G6
    c = load_config(a.config); cset(c.model, "variant", "cf_physics")
    cset(c.optim, "epochs", int(getattr(c.optim, "ema_start_epoch", swap_start)) + 1)
    m = tiny_model(c, dmap).to(dev)
    tr = Trainer(m, c, train_df=None, device=dev)
    dl = make_loader(small.head(32), c, dmap, root, train=True, batch_size=8)
    raw0 = {k: v.detach().clone() for k, v in m.state_dict().items()}
    for ep in range(c.optim.epochs):
        tr._epoch(dl, ep)
    ema = tr._ema_state_dict()
    diff = max(float((ema[k].float() - m.state_dict()[k].float()).abs().max())
               for k in ema if ema[k].dtype.is_floating_point) if ema else -1
    gate("G6 EMA state exists after ema_start and differs from raw weights",
         ema is not None and tr.ema_updates > 0 and diff > 0,
         f"updates={tr.ema_updates} max|ema-raw|={diff:.2e} beta={tr.ema_beta}")

    # ---------------------------------------------------------------- G7
    c = load_config(a.config); cset(c.model, "variant", "casg_lite")
    cset(c.casg, "lambda_cf_ab", 0.5); cset(c.casg, "lambda_cf_as", 0.25)
    m = tiny_model(c, dmap).to(dev); tr = Trainer(m, c, train_df=None, device=dev)
    st = TS._ensure_state(tr)
    gate("G7 consistency weights read from config", st["w_ab"] == 0.5 and st["w_as"] == 0.25,
         f"w_ab={st['w_ab']} w_as={st['w_as']}")

    print()
    if FAILS:
        print(f"*** {len(FAILS)} GATE(S) FAILED: {FAILS}\nDo not launch."); sys.exit(1)
    print("ALL V2 GATES PASSED — launch with: bash scripts/v2_launch.sh")