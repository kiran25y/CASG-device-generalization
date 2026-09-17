"""CASG training step, protocol v3 + v2-experiment additions (Sept 2026).

VARIANTS (model.variant in the config; the trainer dispatches by method
name "casg" and reads the variant from cfg.model.variant):

  cf_physics     L_cls(a) + w_ab * SKL_sg(p_a, p_b)
                 two physics acquisition views, no third branch.

  casg_identity  cf_physics + a THIRD branch whose patch grid is the anchor's
                 own grid, returned unchanged (explicit identity, P_s = P).
                 The branch still traverses the late encoder with its own
                 dropout realisation, receives classification supervision,
                 and enters the a<->s consistency term. Branch activation
                 follows the SAME donor-eligibility schedule as casg_lite.
                 This is the matched control that isolates the transfer
                 operation from the extra branch and its losses.

  casg_lite      cf_physics + third branch = residual frequency-aware AdaIN
                 toward a RANDOM valid same-class, different-COHORT donor from
                 the batch. Anchors with no eligible donor take the explicit
                 identity path (never detached self-statistics).

  casg_full      casg_lite + style encoder, queue, far-style mining, CSPC,
                 virtual style. Unchanged; not part of the v2 experiment.

Loss (eligible transfer batches):
  L = 0.5 [L_cls(a) + L_cls(s)] + w_ab C(a,b) + w_as C(a,s)
  with w_ab = casg.lambda_cf_ab, w_as = casg.lambda_cf_as (both default to
  casg.lambda_cf so historical configs are unchanged).

Hard rules:
  * label_4class < 0 is never clamped and never used for donor matching.
  * donors come from a different COHORT (physical patient); the dataset
    emits batch["pid"] as a stable hash of cohort_uid.
  * ineligible anchors use an explicit identity path, not detached
    self-statistics (forward-equal but gradient-different).
  * per-epoch diagnostics are accumulated in st["log"] and drained by the
    trainer (casg_epoch_log) so the archive records what actually happened.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from src.losses import pathology_loss, supcon_loss
from src.models.acpl import cf_consistency          # symmetric stop-grad KL
from .style_encoder import style_info_nce
from .style_queue import StyleQueue
from .adain import token_stats, residual_freq_adain, interpolate_stats

VARIANTS = ("cf_physics", "casg_identity", "casg_lite", "casg_full")
_INTERVENTION = ("casg_identity", "casg_lite", "casg_full")


def _cfg_get(c, key, default):
    if c is None:
        return default
    if isinstance(c, dict):
        return c.get(key, default)
    return getattr(c, key, default)


def _phase(ep1: int, variant: str, cur) -> int:
    """Variant-aware curriculum. Identity follows casg_lite exactly."""
    if ep1 < int(_cfg_get(cur, "random_swap_start", 6)):
        return 1
    if variant == "cf_physics":
        return 1
    if ep1 < int(_cfg_get(cur, "far_style_start", 13)) or \
       variant in ("casg_lite", "casg_identity"):
        return 2
    if ep1 < int(_cfg_get(cur, "virtual_style_start", 21)):
        return 3
    return 4


def _fresh_log():
    return {"steps": 0, "phase2_steps": 0, "branch_active_steps": 0,
            "anchors": 0, "anchors_valid4": 0, "anchors_with_donor": 0,
            "donor_cross_domain": 0, "fallback_batches": 0,
            "L_cls": 0.0, "L_cf_ab": 0.0, "L_cf_as": 0.0,
            "disp_sum": 0.0, "disp_max": 0.0, "disp_n": 0}


def _ensure_state(tr):
    if not hasattr(tr, "_casg"):
        c = getattr(tr.cfg, "casg", None)
        variant = str(_cfg_get(getattr(tr.cfg, "model", None), "variant",
                               "casg_lite")).lower()
        if variant not in VARIANTS:
            raise ValueError(f"model.variant={variant!r} not in {VARIANTS}")
        l_cf = float(_cfg_get(c, "lambda_cf", 1.0))
        tr._casg = {
            "variant": variant,
            "queue": StyleQueue(capacity=int(_cfg_get(c, "style_queue_per_class", 256))),
            "rho": float(_cfg_get(c, "residual_style_strength", 0.5)),
            "w_ab": float(_cfg_get(c, "lambda_cf_ab", l_cf)),
            "w_as": float(_cfg_get(c, "lambda_cf_as", l_cf)),
            "l_style": float(_cfg_get(c, "lambda_style", 0.10)),
            "l_cspc": float(_cfg_get(c, "lambda_cspc", 0.20)),
            "tau_s": float(_cfg_get(c, "style_temperature", 0.07)),
            "cur": _cfg_get(c, "curriculum", None),
            "fallback": [0, 0],
            "logged": False,
            "log": _fresh_log(),
        }
    return tr._casg


def _valid4(y4: torch.Tensor) -> torch.Tensor:
    return y4 >= 0


def _batch_donor_index(y4: torch.Tensor, pid: torch.Tensor,
                       gen: torch.Generator = None) -> torch.Tensor:
    """Random valid same-class, DIFFERENT-cohort row. -1 when none exists."""
    B = y4.shape[0]
    out = torch.full((B,), -1, dtype=torch.long)
    valid = _valid4(y4)
    for i in range(B):
        if not bool(valid[i]):
            continue
        cand = ((y4 == y4[i]) & valid & (pid != pid[i])).nonzero(as_tuple=True)[0]
        if cand.numel():
            k = int(torch.randint(0, cand.numel(), (1,), generator=gen).item())
            out[i] = cand[k]
    return out


def _rel_disp(p_s: torch.Tensor, p: torch.Tensor, eps: float = 1e-6):
    """r_i = ||P_s - P||_F / (||P||_F + eps), per anchor, on the patch grid."""
    d = (p_s - p).flatten(1).norm(dim=1)
    n = p.flatten(1).norm(dim=1)
    return d / (n + eps)


def casg_step(tr, batch, ep):
    st = _ensure_state(tr)
    model, dev = tr.model, tr.device
    variant = st["variant"]
    phase = _phase(int(ep) + 1, variant, st["cur"])
    log = st["log"]

    mel_a = batch["mel"].to(dev)
    mel_b = batch["mel2"].to(dev)
    y4 = batch["label4"].to(dev)
    y2 = batch["label2"].to(dev)
    pid = batch["pid"].to(dev)
    dev_id = batch.get("device_id")
    dev_id = dev_id.to(dev) if dev_id is not None else None
    B = mel_a.shape[0]

    # ---- task forwards: both acquisition views carry gradient ------------
    h9a, taps_a = model.forward_early(mel_a)
    h9b, _ = model.forward_early(mel_b)

    grid = model.patch_grid(h9a)                    # [B, F, T, H]
    mu_i, ls_i = token_stats(grid.detach())
    views = [h9a, h9b]
    n_swap_views = 0
    donor_mu = donor_ls = None
    have = torch.zeros(B, dtype=torch.bool)

    log["steps"] += 1
    log["anchors"] += B
    log["anchors_valid4"] += int(_valid4(y4).sum())

    if variant in _INTERVENTION and phase >= 2:
        log["phase2_steps"] += 1
        st["fallback"][1] += 1

        if variant in ("casg_lite", "casg_identity"):
            # identical eligibility schedule for lite and identity
            idx = _batch_donor_index(y4.cpu(), pid.cpu())
            have = idx >= 0
            if bool(have.any()):
                src = idx.clamp_min(0)
                donor_mu = mu_i[src].clone()
                donor_ls = ls_i[src].clone()
                log["anchors_with_donor"] += int(have.sum())
                if dev_id is not None:
                    cross = (dev_id[src.to(dev)] != dev_id).cpu() & have
                    log["donor_cross_domain"] += int(cross.sum())
            else:
                st["fallback"][0] += 1
                log["fallback_batches"] += 1
        else:                                        # casg_full: learned queue
            s_anchor = model.style_enc(
                {k: v.detach() for k, v in taps_a.items()},
                batch.get("mel_clean", mel_a).to(dev), None)
            donor_mu, donor_ls = mu_i.clone(), ls_i.clone()
            for i in range(B):
                if not bool(_valid4(y4[i])):
                    continue
                it = st["queue"].sample(int(y4[i]), s_anchor[i], int(pid[i]),
                                        hard=(phase >= 3))
                if it is not None:
                    donor_mu[i] = it["mu"].to(dev)
                    donor_ls[i] = it["ls"].to(dev)
                    have[i] = True
            if not bool(have.any()):
                st["fallback"][0] += 1
                log["fallback_batches"] += 1
                donor_mu = donor_ls = None

        if donor_mu is not None:
            log["branch_active_steps"] += 1
            have_d = have.to(dev)[:, None, None, None]
            if variant == "casg_identity":
                # EXPLICIT identity: the anchor's own grid, unchanged.
                swapped = grid
            else:
                swapped = residual_freq_adain(grid, donor_mu, donor_ls, st["rho"])
                # anchors without a donor take the explicit identity path
                swapped = torch.where(have_d, swapped, grid)
            with torch.no_grad():
                r = _rel_disp(swapped.detach(), grid.detach())[have.to(dev)]
                if r.numel():
                    log["disp_sum"] += float(r.sum()); log["disp_n"] += int(r.numel())
                    log["disp_max"] = max(log["disp_max"], float(r.max()))
            views.append(model.merge_grid(h9a, swapped))
            n_swap_views += 1
            if variant == "casg_full" and phase >= 4:
                lam = float(torch.distributions.Beta(2.0, 2.0).sample())
                mu_v, ls_v = interpolate_stats(mu_i, ls_i, donor_mu, donor_ls, lam)
                views.append(model.merge_grid(
                    h9a, residual_freq_adain(grid, mu_v, ls_v, st["rho"])))
                n_swap_views += 1

    if not st["logged"]:
        print(f"[casg] variant={variant} phase={phase} views={len(views)} "
              f"rho={st['rho']} w_ab={st['w_ab']} w_as={st['w_as']} "
              f"split={model.split_block}", flush=True)
        st["logged"] = True

    # ---- one batched pass through the shared late blocks -----------------
    # Each view occupies its own rows of the concatenated batch, so dropout
    # in the projector is realised independently per view (R-Drop-style
    # consistency is therefore present in Identity as well as in Lite).
    pooled = model.forward_late(torch.cat(views, dim=0))
    l4, l2, z = model.heads_from_pooled(pooled)
    nv = len(views)
    y4r, y2r = y4.repeat(nv), y2.repeat(nv)

    # classification: anchor + intervention views; NOT the paired view b
    cls_idx = torch.cat([torch.arange(B)] +
                        [torch.arange(k * B, (k + 1) * B) for k in range(2, nv)])
    L_cls = pathology_loss(l4[cls_idx], l2[cls_idx], y4r[cls_idx], y2r[cls_idx],
                           tr.gamma, tr.w4, tr.w2,
                           getattr(tr, "adj4", None), getattr(tr, "adj2", None))
    loss = L_cls

    o4, o2 = l4[:B], l2[:B]
    L_ab = 0.5 * (cf_consistency(o2, l2[B:2 * B]) + cf_consistency(o4, l4[B:2 * B]))
    loss = loss + st["w_ab"] * L_ab
    L_as = torch.zeros((), device=dev)
    for k in range(2, nv):
        s4, s2 = l4[k * B:(k + 1) * B], l2[k * B:(k + 1) * B]
        L_as = L_as + 0.5 * (cf_consistency(o2, s2) + cf_consistency(o4, s4))
    if nv > 2:
        loss = loss + st["w_as"] * L_as

    log["L_cls"] += float(L_cls.detach())
    log["L_cf_ab"] += float(L_ab.detach())
    log["L_cf_as"] += float(L_as.detach())

    # ---- Full CASG only ---------------------------------------------------
    if variant == "casg_full":
        mel_clean_a = batch.get("mel_clean", mel_a).to(dev)
        mel_clean_b = batch.get("mel_clean2", mel_b).to(dev)
        with torch.no_grad():
            _, taps_ca = model.forward_early(mel_clean_a)
            _, taps_cb = model.forward_early(mel_clean_b)
        t_tok = (h9a.shape[1] - model.n_special) // model.f_patches
        half = t_tok // 2
        q = model.style_enc(taps_ca, mel_clean_a, (0, half))
        p = model.style_enc(taps_ca, mel_clean_a, (half, t_tok))
        n = model.style_enc(taps_cb, mel_clean_b, (half, t_tok))
        loss = loss + st["l_style"] * style_info_nce(q, p, n, st["tau_s"])
        if phase >= 3:
            keep = _valid4(y4r)
            if bool(keep.any()):
                loss = loss + st["l_cspc"] * supcon_loss(
                    F.normalize(z[keep], dim=-1), y4r[keep], tr.tau)
        s_full = model.style_enc(taps_ca, mel_clean_a, None)
        for i in range(B):
            if bool(_valid4(y4[i])):
                st["queue"].enqueue(int(y4[i]), int(pid[i]), s_full[i],
                                    mu_i[i], ls_i[i])
    return loss


def casg_epoch_log(tr) -> dict:
    """Drain and return per-epoch diagnostics (called by the trainer)."""
    st = getattr(tr, "_casg", None)
    if st is None:
        return {}
    g = st["log"]; s = max(1, g["steps"])
    out = {
        "variant": st["variant"], "steps": g["steps"],
        "phase2_steps": g["phase2_steps"],
        "branch_active_steps": g["branch_active_steps"],
        "donor_util": g["anchors_with_donor"] / max(1, g["anchors_valid4"]),
        "cross_domain_frac": g["donor_cross_domain"] / max(1, g["anchors_with_donor"]),
        "fallback_batches": g["fallback_batches"],
        "L_cls": g["L_cls"] / s, "L_cf_ab": g["L_cf_ab"] / s, "L_cf_as": g["L_cf_as"] / s,
        "disp_mean": g["disp_sum"] / max(1, g["disp_n"]), "disp_max": g["disp_max"],
    }
    st["log"] = _fresh_log()
    return out


# ------------------------------------------------------------ checkpoint --
def casg_state_dict(tr) -> dict:
    st = getattr(tr, "_casg", None)
    if st is None:
        return {}
    return {"variant": st["variant"], "fallback": list(st["fallback"]),
            "queue": {c: list(d) for c, d in st["queue"].q.items()}}


def load_casg_state(tr, blob: dict) -> None:
    if not blob:
        return
    st = _ensure_state(tr)
    st["fallback"] = list(blob.get("fallback", [0, 0]))
    for c, items in (blob.get("queue") or {}).items():
        st["queue"].q[int(c)].clear()
        for it in items:
            st["queue"].q[int(c)].append(it)