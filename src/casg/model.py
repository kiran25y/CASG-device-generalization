"""CASG-Net (protocol sections 3, 10, 19.1).

Wraps the repo's ASTBackbone and splits the HuggingFace ASTModel after
``split_block`` (default 9): ``forward_early`` runs embeddings + blocks
1..9 and exposes detached-able taps; ``forward_late`` runs blocks 10..12 +
final layernorm + AST pooling for one or several stacked token views.

The style branch (style encoder) lives here for checkpointing but is used
ONLY by the training step. ``forward`` is the deployment path: a single
original-view pass — no style encoder, no queue, no simulator, no transfer.

``verify_split`` is a hard gate: the composed early+late forward must
reproduce the stock ASTModel forward. If HF internals change, this fails
loudly instead of training a silently wrong model.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.backbones import ASTBackbone
from .style_encoder import StyleEncoder


class CASGNet(nn.Module):
    def __init__(self, cfg, n_devices: int = 5):
        super().__init__()
        if str(cfg.model.backbone).lower() != "ast":
            raise ValueError("CASG requires the AST backbone")
        self.backbone = ASTBackbone(cfg)
        self.backbone.head = nn.Identity()            # CASG uses its own projector
        self.ast = self.backbone.net                  # HF ASTModel
        c = getattr(cfg, "casg", {}) or {}
        self.split_block = int(c.get("split_block", 9) if isinstance(c, dict)
                               else getattr(c, "split_block", 9))
        taps = (c.get("style_taps", [3, 6, 9]) if isinstance(c, dict)
                else getattr(c, "style_taps", [3, 6, 9]))
        self.taps = tuple(int(t) for t in taps)
        n_layers = len(self.ast.encoder.layer)
        assert 0 < self.split_block < n_layers, "bad split_block"
        assert all(0 < t <= self.split_block for t in self.taps), \
            "style taps must lie inside the early stack"

        hidden = int(self.ast.config.hidden_size)
        fcfg = self.ast.config
        self.f_patches = (int(fcfg.num_mel_bins) - int(fcfg.patch_size)) \
            // int(fcfg.frequency_stride) + 1
        self.n_special = 2                            # cls + distillation

        pdim = int(c.get("pathology_dim", 256) if isinstance(c, dict)
                   else getattr(c, "pathology_dim", 256))
        self.path_proj = nn.Sequential(nn.Linear(hidden, 512), nn.GELU(),
                                       nn.Dropout(float(cfg.model.dropout)),
                                       nn.Linear(512, pdim))
        self.h4 = nn.Linear(pdim, int(cfg.model.n_classes4))
        self.h2 = nn.Linear(pdim, int(cfg.model.n_classes2))
        sdim = int(c.get("style_dim", 128) if isinstance(c, dict)
                   else getattr(c, "style_dim", 128))
        self.style_enc = StyleEncoder(hidden=hidden,
                                      mel_bins=int(cfg.audio.n_mels),
                                      f_patches=self.f_patches,
                                      n_special=self.n_special,
                                      taps=self.taps, style_dim=sdim)
        self.mixstyle = None                          # trainer prints this attr

    # ------------------------------------------------------------ plumbing
    def _prep(self, mel: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        x = mel.squeeze(1) if mel.dim() == 4 else mel
        x = x.transpose(1, 2).contiguous()            # [B, T, F]
        T = x.shape[1]
        if T != bb.n_frames:
            if T > bb.n_frames:
                x = x[:, :bb.n_frames, :]
            else:
                x = F.pad(x, (0, 0, 0, bb.n_frames - T))
        return x * bb.in_scale

    def forward_early(self, mel: torch.Tensor):
        """-> (h_split [B, N, H], taps {block: tokens})."""
        h = self.ast.embeddings(self._prep(mel))
        out = {}
        for i, layer in enumerate(self.ast.encoder.layer[:self.split_block]):
            o = layer(h)
            h = o[0] if isinstance(o, (tuple, list)) else o
            if (i + 1) in self.taps:
                out[i + 1] = h
        return h, out

    def forward_late(self, h: torch.Tensor) -> torch.Tensor:
        """blocks split+1..12 + layernorm + AST pooling -> pooled hidden."""
        for layer in self.ast.encoder.layer[self.split_block:]:
            o = layer(h)
            h = o[0] if isinstance(o, (tuple, list)) else o
        h = self.ast.layernorm(h)
        return (h[:, 0] + h[:, 1]) / 2.0              # HF AST pooler

    def patch_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, hdim = tokens.shape
        t = (n - self.n_special) // self.f_patches
        return tokens[:, self.n_special:, :].reshape(b, self.f_patches, t, hdim)

    def merge_grid(self, tokens: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        patch = grid.reshape(b, -1, grid.shape[-1])
        return torch.cat([tokens[:, :self.n_special, :], patch], dim=1)

    def heads_from_pooled(self, pooled: torch.Tensor):
        z = self.path_proj(pooled)
        return self.h4(z), self.h2(z), z

    # ---------------------------------------------------------- deployment
    def forward(self, mel, domain=None):
        h, _ = self.forward_early(mel)
        pooled = self.forward_late(h)
        l4, l2, z = self.heads_from_pooled(pooled)
        return {"logits4": l4, "logits2": l2, "feat": z}

    # ------------------------------------------------------------- gates
    @torch.no_grad()
    def verify_split(self, atol: float = 1e-4) -> float:
        """Composed early+late must equal the stock ASTModel forward."""
        was_training = self.training
        self.eval()
        mel = torch.randn(2, 1, int(self.ast.config.num_mel_bins), 101)
        ref = self.ast(input_values=self._prep(mel)).pooler_output
        h, _ = self.forward_early(mel)
        got = self.forward_late(h)
        self.train(was_training)
        if ref.shape != got.shape:
            raise RuntimeError(
                f"AST split-equivalence FAILED: shape mismatch "
                f"ref={tuple(ref.shape)} got={tuple(got.shape)}")
        err = float((ref - got).abs().max())
        if err > atol:
            raise RuntimeError(
                f"AST split-equivalence FAILED (max err {err:.2e}). The HF "
                f"ASTModel internals differ from the assumed layout — do not "
                f"train until this is resolved.")
        return err
