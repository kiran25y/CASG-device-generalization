"""Unified trainer for every method (erm | dann | mixstyle | dmixstyle | coral |
contrastive) on the SAME backbone, so the comparison is apples-to-apples.

Logit-adjusted pathology loss, AMP, warmup->cosine, early stop, per-device
evaluation, and a per-clip prediction dump so that every downstream metric
(AUROC, AUPRC, calibration, per-device, worst-device, bootstrap CIs, ROC/PR
curves, t-SNE) can be recomputed post-hoc without retraining.
"""
from __future__ import annotations
import copy, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from .losses import (pathology_loss, supcon_loss, coral_by_domain,
                     compute_logit_adjust)
from .metrics import compute_metrics, per_device


def resolve_device(cfg):
    d = str(cfg.device)
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(d)


def _lr(ep, warm, total):
    if ep < warm:
        return (ep + 1) / max(1, warm)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (ep - warm) / max(1, total - warm))))


def _dann_lambda(ep, total, lam_max=1.0):
    p = ep / max(1, total - 1)
    return lam_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


class Trainer:
    def __init__(self, model, cfg, train_df=None, device=None, target_device_id=None):
        self.cfg = cfg
        self.device = device or resolve_device(cfg)
        self.model = model.to(self.device)
        self.method = str(getattr(cfg.model, "method", "contrastive")).lower()
        # only used by 'dmixstyle' (the DA oracle row)
        self.target_device_id = target_device_id

        o = cfg.optim
        self.epochs = int(o.epochs)
        self.warm = int(o.warmup_epochs)
        self.baselr = float(o.lr)
        self.clip = float(o.grad_clip)
        self.amp = bool(o.amp) and self.device.type == "cuda"
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.baselr,
                                     weight_decay=float(o.weight_decay))
        try:                                        # torch>=2.4 spelling
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        except (AttributeError, TypeError):         # older torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        # ---- v2: fixed EMA of weights, active from epoch index ema_start ----
        _c = getattr(cfg, "casg", None)
        _cur = getattr(_c, "curriculum", None) if _c is not None else None
        _swap = int(getattr(_cur, "random_swap_start", 6)) if _cur is not None else 6
        self.ema_enabled = bool(getattr(o, "ema", True))
        self.ema_start = int(getattr(o, "ema_start_epoch", _swap)) - 1   # 0-indexed
        self.ema_halflife_epochs = float(getattr(o, "ema_halflife_epochs", 1.0))
        self.ema_state, self.ema_beta, self.ema_updates = None, None, 0

        L = cfg.loss
        self.gamma = float(L.focal_gamma)
        self.w4 = float(L.w_logits4)
        self.w2 = float(L.w_logits2)
        self.supw = float(getattr(L, "supcon_weight", 0.5))
        self.tau = float(getattr(L, "supcon_tau", 0.1))
        self.coralw = float(getattr(L, "coral_weight", 1.0))
        self.coral_min = int(getattr(L, "coral_min_per_domain", 4))
        self.lammax = float(getattr(o, "dann_lambda_max", 1.0))
        self.acpl_lacq = float(getattr(L, "acpl_lambda_acq", 0.5))
        self.acpl_lcf = float(getattr(L, "acpl_lambda_cf", 1.0))

        # ---- stale-code guards. A trainer that predates a method treats it
        # as ERM and trains silently under the wrong label; that once burned
        # 30 GPU-cells of "acpl" runs that were actually ERM. Fail loudly.
        _KNOWN = {"casg", "cfsc", "cfc", "erm2view", "erm", "dann", "coral", "mixstyle", "dmixstyle",
                  "contrastive", "acpl"}
        if self.method not in _KNOWN:
            raise ValueError(f"unknown method '{self.method}' — this trainer "
                             f"knows {sorted(_KNOWN)}; a stale trainer.py "
                             f"would have trained this as ERM silently")
        if self.method == "acpl" and not hasattr(self.model, "acpl_forward"):
            raise TypeError("method=acpl but the model has no acpl_forward — "
                            "the model was built by a stale "
                            "train_contrastive.py as plain DeviceAgnosticNet. "
                            "Sync src/models/acpl.py AND "
                            "scripts/train_contrastive.py, then rerun "
                            "scripts/acpl_smoke.py until it passes.")

        self.adj4 = self.adj2 = None
        if bool(getattr(L, "logit_adjust", False)) and train_df is not None:
            t = float(L.logit_adjust_tau)
            c4 = np.bincount(train_df.loc[train_df.label4 >= 0, "label4"].to_numpy(),
                             minlength=4)
            c2 = np.bincount(train_df["label2"].to_numpy(), minlength=2)
            self.adj4 = compute_logit_adjust(c4, t).to(self.device).unsqueeze(0)
            self.adj2 = compute_logit_adjust(c2, t).to(self.device).unsqueeze(0)
            print(f"[trainer] method={self.method} logit-adjust "
                  f"c4={c4.tolist()} c2={c2.tolist()}", flush=True)

    def _autocast(self):
        try:
            return torch.amp.autocast("cuda", enabled=self.amp)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=self.amp)

    def _step(self, batch, ep):
        if self.method == "casg":
            from src.casg.trainer_step import casg_step
            return casg_step(self, batch, ep)
        if self.method in ("acpl", "cfsc"):
            return self._step_acpl(batch)
        dev = self.device
        mel = batch["mel"].to(dev)
        y4 = batch["label4"].to(dev)
        y2 = batch["label2"].to(dev)
        did = batch["device_id"].to(dev)

        # directional MixStyle needs to know which samples are target-domain
        domain = None
        if self.method == "dmixstyle" and self.target_device_id is not None:
            domain = (did == int(self.target_device_id))

        f1 = self.model.features(mel, domain)
        l4 = self.model.h4(f1)
        l2 = self.model.h2(f1)
        loss = pathology_loss(l4, l2, y4, y2, self.gamma, self.w4, self.w2,
                              self.adj4, self.adj2)

        if self.method == "dann":
            lam = _dann_lambda(ep, self.epochs, self.lammax)
            valid = did >= 0
            if valid.any():
                loss = loss + F.cross_entropy(
                    self.model.device_logits(f1[valid], lam), did[valid])

        elif self.method == "coral":
            # group by ACTUAL device, not by an arbitrary split of the batch
            loss = loss + self.coralw * coral_by_domain(f1, did, self.coral_min)

        elif self.method == "contrastive" and "mel2" in batch:
            f2 = self.model.features(batch["mel2"].to(dev))
            z1 = F.normalize(self.model.proj(f1), dim=-1)
            z2 = F.normalize(self.model.proj(f2), dim=-1)
            # group by 4-class where available; binary-only rows form their own groups
            grp = torch.where(y4 >= 0, y4, 10 + y2)
            loss = loss + self.supw * supcon_loss(torch.cat([z1, z2]),
                                                  torch.cat([grp, grp]), self.tau)

        elif self.method == "cfc":
            # Counterfactual consistency on the PLAIN architecture: two-view
            # averaged classification loss + symmetric stop-gradient KL, no
            # acquisition encoder / FiLM / theta anywhere. If this matches the
            # CF-only ACPL arm externally, the conditioning machinery is
            # unnecessary and the method simplifies to consistency training
            # under synthetic acquisition intervention.
            from src.models.acpl import cf_consistency
            if "mel2" not in batch:
                raise KeyError("cfc batch lacks mel2 — the loader must be "
                               "built with contrastive=True (two views)")
            f2 = self.model.features(batch["mel2"].to(dev))
            l4b, l2b = self.model.h4(f2), self.model.h2(f2)
            loss = 0.5 * (loss + pathology_loss(l4b, l2b, y4, y2, self.gamma,
                                                self.w4, self.w2,
                                                self.adj4, self.adj2))
            loss = loss + self.acpl_lcf * (cf_consistency(l2, l2b) +
                                           cf_consistency(l4, l4b))

        elif self.method == "erm2view":
            # CONTROL for the ACPL ablation: plain architecture, but the
            # classification loss is averaged over the same two synthetic
            # acquisition views ACPL trains on. Isolates the two-view-
            # averaging effect from the conditioning architecture.
            if "mel2" not in batch:
                raise KeyError("erm2view batch lacks mel2 — the loader must "
                               "be built with contrastive=True (two views)")
            f2 = self.model.features(batch["mel2"].to(dev))
            loss = 0.5 * (loss + pathology_loss(
                self.model.h4(f2), self.model.h2(f2), y4, y2, self.gamma,
                self.w4, self.w2, self.adj4, self.adj2))

        # erm / mixstyle / dmixstyle: pathology loss only (MixStyle acts on the input)
        return loss

    def _step_acpl(self, batch):
        """ACPL-v1: L_cls + lambda_acq * L_acq + lambda_CF * L_CF (report Sec. 5.1).
        Ablations ACPL-A/B/full are selected purely by the two weights."""
        from src.models.acpl import acq_loss, cf_consistency, cf_consistency_anchored
        dev = self.device
        if "theta" not in batch or "mel2" not in batch:
            raise KeyError("acpl batch lacks theta/mel2 — src/data/dataset.py "
                           "(and src/data/augment.py) on this machine are "
                           "stale. Sync them and rerun scripts/acpl_smoke.py.")
        y4 = batch["label4"].to(dev)
        y2 = batch["label2"].to(dev)
        o1 = self.model.acpl_forward(batch["mel"].to(dev))
        o2 = self.model.acpl_forward(batch["mel2"].to(dev))

        loss = 0.5 * (
            pathology_loss(o1["logits4"], o1["logits2"], y4, y2, self.gamma,
                           self.w4, self.w2, self.adj4, self.adj2) +
            pathology_loss(o2["logits4"], o2["logits2"], y4, y2, self.gamma,
                           self.w4, self.w2, self.adj4, self.adj2))

        if self.method == "cfsc":
            import torch.nn.functional as _F
            from src.losses import supcon_loss as _sc
            z1 = _F.normalize(self.model.proj(o1["feat"]), dim=-1)
            z2 = _F.normalize(self.model.proj(o2["feat"]), dim=-1)
            grp = torch.where(y4 >= 0, y4, 10 + y2)
            loss = loss + self.supw * _sc(torch.cat([z1, z2]),
                                          torch.cat([grp, grp]), self.tau)
        if self.method != "cfsc" and self.acpl_lacq > 0:
            loss = loss + self.acpl_lacq * 0.5 * (
                acq_loss(o1["theta_hat"], batch["theta"].to(dev)) +
                acq_loss(o2["theta_hat"], batch["theta2"].to(dev)))

        if self.acpl_lcf > 0:
            if "mel0" in batch:                       # literal clean-anchored form
                o0 = self.model.acpl_forward(batch["mel0"].to(dev))
                cf = 0.5 * (cf_consistency_anchored(o0["logits2"], o1["logits2"]) +
                            cf_consistency_anchored(o0["logits2"], o2["logits2"]))
            else:                                     # symmetric two-view form
                cf = cf_consistency(o1["logits2"], o2["logits2"]) + \
                     cf_consistency(o1["logits4"], o2["logits4"])
            loss = loss + self.acpl_lcf * cf
        return loss

    def _epoch(self, loader, ep):
        _ds = getattr(loader, 'dataset', None)
        if _ds is not None and hasattr(_ds, 'casg_epoch'):
            _ds.casg_epoch = int(ep)   # v3 deterministic RNG key
        self.model.train()
        for pg in self.opt.param_groups:
            pg["lr"] = self.baselr * _lr(ep, self.warm, self.epochs)
        tot, n, nb, t0, skipped = 0.0, 0, len(loader), time.time(), 0
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
                self._ema_update()
            tot += float(loss.item()) * b["mel"].size(0)
            n += b["mel"].size(0)
            if i % 20 == 0 or i + 1 == nb:
                print(f"\r  ep {ep:3d} | {i+1}/{nb} | loss {tot/max(1,n):.4f} | "
                      f"{(i+1)/(time.time()-t0):.1f} it/s", end="", flush=True)
        print("", flush=True)
        if skipped:
            print(f"  [warn] {skipped}/{nb} batches had non-finite loss and were "
                  f"SKIPPED (try optim.amp:false or a lower lr)", flush=True)
        return tot / max(1, n)

    @torch.no_grad()
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
        """EMA weights cast back to the model's dtypes (or None)."""
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
    def predict(self, loader):
        """Per-clip probabilities + labels. Everything else is derived from this."""
        self.model.eval()
        P4, P2, T4, T2, DV, IX = [], [], [], [], [], []
        for b in loader:
            out = self.model(b["mel"].to(self.device))
            P4.append(F.softmax(out["logits4"].float(), -1).cpu().numpy())
            P2.append(F.softmax(out["logits2"].float(), -1).cpu().numpy())
            T4.append(b["label4"].numpy())
            T2.append(b["label2"].numpy())
            DV.append(b["device_id"].numpy())
            IX.append(b["index"].numpy())
        cat = lambda xs: np.concatenate(xs) if xs else np.array([])
        return {"prob4": cat(P4), "prob2": cat(P2), "y4": cat(T4), "y2": cat(T2),
                "device_id": cat(DV), "index": cat(IX)}

    def evaluate(self, loader, inv_device=None, device_names=None):
        r = self.predict(loader)
        p4 = r["prob4"].argmax(1)
        p2 = r["prob2"].argmax(1)
        m = compute_metrics(r["y4"], p4, r["y2"], p2, r["prob4"], r["prob2"])
        if device_names is not None:
            names = np.asarray(device_names)
        elif inv_device is not None:
            names = np.array([inv_device.get(int(x), str(int(x)))
                              for x in r["device_id"]])
        else:
            names = None
        if names is not None:
            m["_per_device"] = per_device(names, r["y4"], p4, r["y2"], p2,
                                          r["prob4"], r["prob2"])
        m["_raw"] = r
        return m

    def _save_resume(self, path, ep, best, best_state, pat):
        if not path:
            return
        tmp = path + ".tmp"
        torch.save({"epoch": ep, "model": self.model.state_dict(),
                    "opt": self.opt.state_dict(), "scaler": self.scaler.state_dict(),
                    "best": best, "best_state": best_state, "pat": pat}, tmp)
        os.replace(tmp, path)                      # atomic

    def fit(self, tl, vl=None, resume_path=None):
        best, best_state, pat, start = -1.0, None, 0, 0
        maxp = int(self.cfg.optim.early_stop_patience)
        sel = str(getattr(self.cfg.optim, "select_metric", "auroc2"))

        if resume_path and os.path.exists(resume_path):
            try:
                ck = torch.load(resume_path, map_location=self.device)
                self.model.load_state_dict(ck["model"])
                self.opt.load_state_dict(ck["opt"])
                try:
                    self.scaler.load_state_dict(ck["scaler"])
                except Exception:
                    pass
                start = int(ck["epoch"]) + 1
                best, best_state, pat = ck["best"], ck["best_state"], ck["pat"]
                print(f"[trainer] RESUME from epoch {start} (best={best:.4f})", flush=True)
            except Exception as e:
                print(f"[trainer] resume checkpoint unreadable ({e}); starting fresh",
                      flush=True)

        print(f"[trainer] method={self.method} device={self.device} "
              f"epochs={self.epochs} batches={len(tl)} start_epoch={start} "
              f"select={sel}", flush=True)

        best_ep, best_ema, best_ema_state, best_ema_ep = -1, -1.0, None, -1
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
            print(msg, flush=True)
            self._save_resume(resume_path, ep, best, best_state, pat)
            if vl is not None and pat >= maxp:
                print(f"[trainer] early stop @ {ep}", flush=True)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if resume_path and os.path.exists(resume_path):
            os.remove(resume_path)
        return {"best_val": best, "best_epoch": best_ep,
                "state_dict": best_state or copy.deepcopy(self.model.state_dict()),
                "ema_best_val": best_ema, "ema_best_epoch": best_ema_ep,
                "ema_state_dict": best_ema_state, "ema_updates": int(self.ema_updates),
                "ema_beta": self.ema_beta, "ema_start_epoch": int(self.ema_start),
                "val_curve": val_curve, "diagnostics": diag}
