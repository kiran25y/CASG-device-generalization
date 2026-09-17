"""v2 experiment: apply the corrected common pipeline to the existing code.

    python scripts/v2_patch.py            # apply
    python scripts/v2_patch.py --check    # report only, change nothing

Every edit is an EXACT-STRING replacement. If a pattern is not found the
script aborts and prints the surrounding file region so the mismatch can be
resolved by hand; nothing is guessed. Re-running is safe: edits already
present are skipped.

Edits
  1. src/data/augment.py     SpecAugment.paired(ma, mb): one mask set sampled,
                             applied to both views (per-view mean fill).
  2. src/data/dataset.py     (a) casg two-view path uses spec.paired
                             (b) batch["pid"] = stable hash of cohort_uid
                             (c) cohort_uid added in __init__ if absent
                             (d) sample uid emitted as batch["uid"] string
  3. src/trainer.py          fixed EMA of weights from epoch index
                             (random_swap_start-1), updated only after
                             successful optimizer steps; best raw AND best
                             EMA checkpoints tracked on source validation;
                             per-epoch val curve and casg diagnostics kept.
  4. scripts/train_contrastive.py
                             save raw + EMA checkpoints/preds, uid column in
                             preds, selected epochs + val curves + diagnostics
                             in metrics json.
  5. configs/casg_identity.yaml generated from casg_lite.yaml.
"""
import io, os, re, sys, shutil

CHECK = "--check" in sys.argv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def _read(p): return io.open(p, encoding="utf-8").read()
def _write(p, s):
    if CHECK: return
    shutil.copy(p, p + ".v2bak")
    io.open(p, "w", encoding="utf-8").write(s)


def replace_once(text, old, new, label, path):
    if new in text and old not in text:
        print(f"  = already applied: {label}"); return text
    n = text.count(old)
    if n != 1:
        i = text.find(old[:40])
        ctx = text[max(0, i - 300):i + 600] if i >= 0 else "(first 40 chars not found)"
        print(f"\n*** PATTERN {'MISSING' if n == 0 else 'AMBIGUOUS (' + str(n) + ')'}: "
              f"{label} in {path}\n--- expected ---\n{old}\n--- context ---\n{ctx}\n")
        sys.exit(1)
    print(f"  + {label}")
    return text.replace(old, new, 1)


# ============================================================ 1. augment.py
p = "src/data/augment.py"; t = _read(p)
t = replace_once(t,
"""    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.clone()
        nm, nt = mel.shape[-2], mel.shape[-1]""",
"""    def sample_masks(self, nm: int, nt: int):
        \"\"\"Draw one mask set (positions + widths) without applying it.\"\"\"
        ms = []
        for _ in range(self.n):
            f0 = fw = t0 = tw = 0
            if self.f > 0 and nm > 1:
                fw = int(torch.randint(0, min(self.f, nm) + 1, (1,)).item())
                if fw:
                    f0 = int(torch.randint(0, nm - fw + 1, (1,)).item())
            if self.t > 0 and nt > 1:
                tw = int(torch.randint(0, min(self.t, nt) + 1, (1,)).item())
                if tw:
                    t0 = int(torch.randint(0, nt - tw + 1, (1,)).item())
            ms.append((f0, fw, t0, tw))
        return ms

    def apply_masks(self, mel: torch.Tensor, ms) -> torch.Tensor:
        mel = mel.clone()
        v = mel.mean() if self.fill == "mean" else 0.0
        for f0, fw, t0, tw in ms:
            if fw:
                mel[..., f0:f0 + fw, :] = v
            if tw:
                mel[..., :, t0:t0 + tw] = v
        return mel

    def paired(self, ma: torch.Tensor, mb: torch.Tensor):
        \"\"\"SHARED mask positions/widths for two paired views; each view is
        filled with its own mean. v2 corrected pipeline.\"\"\"
        assert ma.shape == mb.shape, "paired views must share a shape"
        ms = self.sample_masks(ma.shape[-2], ma.shape[-1])
        return self.apply_masks(ma, ms), self.apply_masks(mb, ms)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.clone()
        nm, nt = mel.shape[-2], mel.shape[-1]""", "SpecAugment.paired", p)
_write(p, t)

# ============================================================ 2. dataset.py
p = "src/data/dataset.py"; t = _read(p)
# (a) shared masks on the live casg two-view path
t = replace_once(t,
"                m1, m2 = self.spec(m1), self.spec(m2)",
"                m1, m2 = self.spec.paired(m1, m2)      # v2: shared mask positions",
"casg path uses spec.paired", p)
# (b) pid from cohort_uid via stable hash; (d) uid string
m = re.search(r'^(\s*)out\["pid"\]\s*=\s*(.+)$', t, flags=re.M)
if not m:
    print("\n*** could not locate the out[\"pid\"] assignment in dataset.py; paste that line.")
    sys.exit(1)
old_pid = m.group(0); ind = m.group(1)
new_pid = (f'{ind}out["pid"] = torch.tensor(_stable_hash(str(r["cohort_uid"])), dtype=torch.long)  # v2: COHORT key\n'
           f'{ind}out["uid"] = str(r.get("sample_uid", r.get("filepath", i)))')
if "_stable_hash(str(r[\"cohort_uid\"]))" not in t:
    print(f"  + pid -> cohort_uid hash  (was: {old_pid.strip()})")
    t = t.replace(old_pid, new_pid, 1)
else:
    print("  = already applied: pid -> cohort_uid hash")
# helper
t = replace_once(t,
"def _fix(w, L, train, mode=\"repeat\"):",
"""def _stable_hash(s: str) -> int:
    \"\"\"Process-independent 63-bit hash (Python's hash() is salted per run).\"\"\"
    import hashlib
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") >> 1


def _fix(w, L, train, mode="repeat"):""", "_stable_hash helper", p)
# (c) cohort_uid guaranteed
t = replace_once(t,
"        self.df = df.reset_index(drop=True)",
"""        self.df = df.reset_index(drop=True)
        if "cohort_uid" not in self.df.columns:                 # v2
            from . import manifest as _M
            self.df = _M.add_cohort_uid(_M.add_patient_uid(self.df))""",
"cohort_uid in __init__", p)
_write(p, t)

# ============================================================ 3. trainer.py
p = "src/trainer.py"; t = _read(p)
t = replace_once(t,
"""        L = cfg.loss
        self.gamma = float(L.focal_gamma)""",
"""        # ---- v2: fixed EMA of weights, active from epoch index ema_start ----
        _c = getattr(cfg, "casg", None)
        _cur = getattr(_c, "curriculum", None) if _c is not None else None
        _swap = int(getattr(_cur, "random_swap_start", 6)) if _cur is not None else 6
        self.ema_enabled = bool(getattr(o, "ema", True))
        self.ema_start = int(getattr(o, "ema_start_epoch", _swap)) - 1   # 0-indexed
        self.ema_halflife_epochs = float(getattr(o, "ema_halflife_epochs", 1.0))
        self.ema_state, self.ema_beta, self.ema_updates = None, None, 0

        L = cfg.loss
        self.gamma = float(L.focal_gamma)""", "EMA config", p)

t = replace_once(t,
"""        tot, n, nb, t0, skipped = 0.0, 0, len(loader), time.time(), 0
        for i, b in enumerate(loader):
            self.opt.zero_grad(set_to_none=True)
            with self._autocast():
                loss = self._step(b, ep)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.scaler.step(self.opt)
            self.scaler.update()""",
"""        tot, n, nb, t0, skipped = 0.0, 0, len(loader), time.time(), 0
        if self.ema_enabled and self.ema_beta is None:
            U = max(1, nb)
            self.ema_beta = 2.0 ** (-1.0 / (self.ema_halflife_epochs * U))
        for i, b in enumerate(loader):
            self.opt.zero_grad(set_to_none=True)
            with self._autocast():
                loss = self._step(b, ep)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            _scale_before = float(self.scaler.get_scale()) if self.amp else 1.0
            self.scaler.step(self.opt)
            self.scaler.update()
            _stepped = (not self.amp) or float(self.scaler.get_scale()) >= _scale_before
            if self.ema_enabled and _stepped and ep >= self.ema_start:
                self._ema_update()""", "EMA update in _epoch", p)

t = replace_once(t,
"""    @torch.no_grad()
    def predict(self, loader):""",
"""    @torch.no_grad()
    def _ema_update(self):
        sd = self.model.state_dict()
        if self.ema_state is None:                     # init at first successful step
            self.ema_state = {k: v.detach().clone().float() for k, v in sd.items()}
            self.ema_updates = 1
            return
        b = self.ema_beta
        for k, v in sd.items():
            if v.dtype.is_floating_point:
                self.ema_state[k].mul_(b).add_(v.detach().float(), alpha=1.0 - b)
            else:
                self.ema_state[k] = v.detach().clone()
        self.ema_updates += 1

    def _ema_state_dict(self):
        \"\"\"EMA weights cast back to the model's dtypes (or None).\"\"\"
        if self.ema_state is None:
            return None
        sd = self.model.state_dict()
        return {k: self.ema_state[k].to(sd[k].dtype) for k in sd}

    @torch.no_grad()
    def _evaluate_state(self, loader, state):
        cur = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(state)
        v = self.evaluate(loader)
        self.model.load_state_dict(cur)
        return v

    @torch.no_grad()
    def predict(self, loader):""", "EMA helpers", p)

t = replace_once(t,
"""        for ep in range(start, self.epochs):
            loss = self._epoch(tl, ep)
            msg = f"[trainer] ep {ep:3d}/{self.epochs} loss={loss:.4f}"
            if vl is not None:
                v = self.evaluate(vl)
                # model selection on a THRESHOLD-FREE metric: argmax-based scores
                # move with the class prior and pick badly-calibrated epochs.
                s = v.get(sel, float("nan"))
                if not np.isfinite(s):
                    s = v["icbhi2"] if not np.isfinite(v["icbhi4"]) else v["icbhi4"]
                msg += (f" | val auroc2={v['auroc2']:.4f} icbhi4={v['icbhi4']:.4f} "
                        f"icbhi2={v['icbhi2']:.4f}")
                if s > best:
                    best, pat = s, 0
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    pat += 1
            print(msg, flush=True)""",
"""        best_ep, best_ema, best_ema_state, best_ema_ep = -1, -1.0, None, -1
        val_curve, diag = [], []
        for ep in range(start, self.epochs):
            loss = self._epoch(tl, ep)
            msg = f"[trainer] ep {ep:3d}/{self.epochs} loss={loss:.4f}"
            try:                                            # v2 diagnostics
                from src.casg.trainer_step import casg_epoch_log
                d = casg_epoch_log(self)
            except Exception:
                d = {}
            if d:
                d["epoch"] = int(ep); diag.append(d)
                msg += (f" | donor_util={d['donor_util']:.2f} xdom={d['cross_domain_frac']:.2f}"
                        f" disp={d['disp_mean']:.3f}/{d['disp_max']:.3f}"
                        f" Lcls={d['L_cls']:.3f} Lab={d['L_cf_ab']:.3f} Las={d['L_cf_as']:.3f}")
            if vl is not None:
                v = self.evaluate(vl)
                s = v.get(sel, float("nan"))
                if not np.isfinite(s):
                    s = v["icbhi2"] if not np.isfinite(v["icbhi4"]) else v["icbhi4"]
                rec = {"epoch": int(ep), "loss": float(loss), "val_" + sel: float(s),
                       "val_auroc2": float(v["auroc2"])}
                msg += f" | val auroc2={v['auroc2']:.4f}"
                if s > best:
                    best, pat, best_ep = s, 0, int(ep)
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    pat += 1
                ema_sd = self._ema_state_dict()
                if ema_sd is not None:
                    ve = self._evaluate_state(vl, ema_sd)
                    se = ve.get(sel, float("nan"))
                    rec["val_ema_" + sel] = float(se); rec["ema_updates"] = int(self.ema_updates)
                    msg += f" ema={ve['auroc2']:.4f}"
                    if np.isfinite(se) and se > best_ema:
                        best_ema, best_ema_ep, best_ema_state = float(se), int(ep), ema_sd
                val_curve.append(rec)
            print(msg, flush=True)""", "fit loop: raw + EMA selection", p)

t = replace_once(t,
"""        return {"best_val": best,
                "state_dict": best_state or copy.deepcopy(self.model.state_dict())}""",
"""        return {"best_val": best, "best_epoch": best_ep,
                "state_dict": best_state or copy.deepcopy(self.model.state_dict()),
                "ema_best_val": best_ema, "ema_best_epoch": best_ema_ep,
                "ema_state_dict": best_ema_state, "ema_updates": int(self.ema_updates),
                "ema_beta": self.ema_beta, "ema_start_epoch": int(self.ema_start),
                "val_curve": val_curve, "diagnostics": diag}""", "fit return", p)
_write(p, t)

# ================================================= 4. train_contrastive.py
p = "scripts/train_contrastive.py"; t = _read(p)
t = replace_once(t,
"""    torch.save({"state_dict": tr.model.state_dict(), "config": cfg.to_dict(),
                "device_map": dmap}, os.path.join(out_dir, f"ckpt_{tag}.pt"))""",
"""    torch.save({"state_dict": tr.model.state_dict(), "config": cfg.to_dict(),
                "device_map": dmap, "best_epoch": fit_info.get("best_epoch", -1),
                "selection": "raw"}, os.path.join(out_dir, f"ckpt_{tag}.pt"))
    # ---- v2: EMA checkpoint + predictions (secondary; raw stays primary) ----
    ema_sd = fit_info.get("ema_state_dict")
    if ema_sd is not None:
        torch.save({"state_dict": ema_sd, "config": cfg.to_dict(), "device_map": dmap,
                    "best_epoch": fit_info.get("ema_best_epoch", -1), "selection": "ema",
                    "ema_updates": fit_info.get("ema_updates"), "ema_beta": fit_info.get("ema_beta")},
                   os.path.join(out_dir, f"ckpt_{tag}_ema.pt"))
        _cur = copy.deepcopy(tr.model.state_dict()); tr.model.load_state_dict(ema_sd)
        raw_ema = tr.predict(test_loader); tr.model.load_state_dict(_cur)
        meta_ema = test_df.iloc[raw_ema["index"].astype(int)].reset_index(drop=True)
        np.savez_compressed(
            os.path.join(out_dir, f"preds_{tag}_ema.npz"),
            prob4=raw_ema["prob4"], prob2=raw_ema["prob2"], y4=raw_ema["y4"], y2=raw_ema["y2"],
            device=meta_ema["device"].to_numpy().astype(str),
            patient=meta_ema["patient_id"].to_numpy().astype(str),
            cohort=meta_ema["cohort_uid"].to_numpy().astype(str) if "cohort_uid" in meta_ema else meta_ema["patient_id"].to_numpy().astype(str),
            source=meta_ema["source"].to_numpy().astype(str),
            uid=(meta_ema["sample_uid"] if "sample_uid" in meta_ema else meta_ema["filepath"]).to_numpy().astype(str),
            method=method, backbone=bk, held_out=a.held_out, seed=a.seed, selection="ema")""",
"save EMA ckpt/preds", p)
t = replace_once(t,
"""        patient=test_meta["patient_id"].to_numpy().astype(str),
        source=test_meta["source"].to_numpy().astype(str),
        method=method, backbone=bk, held_out=a.held_out, seed=a.seed)""",
"""        patient=test_meta["patient_id"].to_numpy().astype(str),
        cohort=test_meta["cohort_uid"].to_numpy().astype(str) if "cohort_uid" in test_meta else test_meta["patient_id"].to_numpy().astype(str),
        source=test_meta["source"].to_numpy().astype(str),
        uid=(test_meta["sample_uid"] if "sample_uid" in test_meta else test_meta["filepath"]).to_numpy().astype(str),
        method=method, backbone=bk, held_out=a.held_out, seed=a.seed, selection="raw")""",
"uid/cohort in raw preds", p)
# metrics json: locate the flat.update({...}) block by its opening and insert
# before its closing '})' — the box's version of the dict body differs.
_open = 'flat.update({"method": method, "backbone": bk, "held_out": a.held_out,'
_add = ('''
                 "best_val_score": float(fit_info["best_val"]),
                 "select_metric": str(cfg.optim.select_metric),
                 "best_epoch": fit_info.get("best_epoch", -1),
                 "ema_best_val_score": fit_info.get("ema_best_val"),
                 "ema_best_epoch": fit_info.get("ema_best_epoch", -1),
                 "ema_updates": fit_info.get("ema_updates"), "ema_beta": fit_info.get("ema_beta"),
                 "ema_start_epoch": fit_info.get("ema_start_epoch"),
                 "variant": str(getattr(cfg.model, "variant", "")),
                 "val_curve": fit_info.get("val_curve", []),
                 "diagnostics": fit_info.get("diagnostics", [])''')
if '"diagnostics": fit_info.get("diagnostics", [])' in t:
    print("  = already applied: metrics json: epochs, curves, diagnostics")
else:
    i = t.find(_open)
    if i < 0:
        print(f"\n*** PATTERN MISSING: flat.update opening in {p}"); sys.exit(1)
    j = t.find("})", i)
    if j < 0:
        print(f"\n*** could not find closing '}})' of flat.update in {p}"); sys.exit(1)
    body = t[i:j].rstrip().rstrip(",")
    t = t[:i] + body + "," + _add + t[j:]
    print("  + metrics json: epochs, curves, diagnostics")
t = replace_once(t, "import _boot, argparse, os, json", "import _boot, argparse, os, json, copy",
                 "import copy", p)
_write(p, t)

# ====================================================== 5. identity config
src, dst = "configs/casg_lite.yaml", "configs/casg_identity.yaml"
c = _read(src)
if "variant: casg_lite" not in c:
    print("*** configs/casg_lite.yaml lacks 'variant: casg_lite'"); sys.exit(1)
if not CHECK:
    io.open(dst, "w", encoding="utf-8").write(c.replace("variant: casg_lite", "variant: casg_identity", 1))
print(f"  + {dst}")

print("\nPATCH " + ("CHECK COMPLETE (no files changed)" if CHECK else "APPLIED — backups: *.v2bak"))
print("Next: python scripts/v2_gates.py   (must print ALL V2 GATES PASSED)")