"""Backbones: AST (AudioSet-pretrained), timm (ImageNet-pretrained), light CNN.

Note on the previous version
-----------------------------
`configs/*.yaml` and the README both advertised ``timm_name: ast``, but the
implementation just called ``timm.create_model(name)`` and **timm has no AST
model** — that setting would have raised. Worse, `timm_pretrained: true` gives
ImageNet weights, whereas the reference work on this cohort uses AST with
*AudioSet* pretraining, which is the dominant lever in respiratory-sound
classification. AST is therefore implemented properly here, from HuggingFace.

MixStyle is deliberately NOT applied inside these modules: it lives at the
log-mel input in `net.py` so that it applies identically to every backbone.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# AST — Audio Spectrogram Transformer, AudioSet-pretrained
# ---------------------------------------------------------------------------
class ASTBackbone(nn.Module):
    """HuggingFace AST with the positional embedding re-gridded to our clip
    length.

    The pretrained checkpoint is built for 1024 frames (10.24 s). Our clips are
    ``clip_seconds`` long, which at a 10 ms hop gives far fewer frames, so the
    time axis of the pretrained position grid is interpolated down rather than
    re-initialised — re-initialising would throw away most of the value of
    AudioSet pretraining.
    """

    def __init__(self, cfg):
        super().__init__()
        try:
            from transformers import ASTModel
        except Exception as exc:                                # pragma: no cover
            raise RuntimeError(
                "backbone 'ast' needs HuggingFace transformers:\n"
                "    pip install 'transformers>=4.35'\n"
                f"(import failed: {exc})") from exc

        name = str(getattr(cfg.model, "ast_name",
                           "MIT/ast-finetuned-audioset-10-10-0.4593"))
        pretrained = bool(getattr(cfg.model, "timm_pretrained", True))

        n_mels = int(cfg.audio.n_mels)
        sr = int(cfg.audio.sample_rate)
        hop = int(cfg.audio.hop_length)
        # torchaudio MelSpectrogram is center=True -> 1 + L//hop frames
        self.n_frames = 1 + int(sr * float(cfg.audio.clip_seconds)) // hop

        if pretrained:
            self.net = ASTModel.from_pretrained(name)
        else:
            from transformers import ASTConfig
            self.net = ASTModel(ASTConfig(num_mel_bins=n_mels,
                                          max_length=self.n_frames))

        if int(self.net.config.num_mel_bins) != n_mels:
            raise ValueError(
                f"AST expects num_mel_bins={self.net.config.num_mel_bins} but "
                f"audio.n_mels={n_mels}. Set audio.n_mels to "
                f"{self.net.config.num_mel_bins}.")

        self._regrid_position_embeddings(self.n_frames)
        self.net.config.max_length = self.n_frames

        # AST was trained on features normalised to roughly zero mean / 0.5 std
        # (HF ASTFeatureExtractor: (x-mean)/(2*std)). Our MelExtractor produces
        # unit variance, so rescale to land in the distribution the pretrained
        # weights expect.
        self.in_scale = float(getattr(cfg.model, "ast_input_scale", 0.5))

        hidden = int(self.net.config.hidden_size)
        self.feat_dim = int(cfg.model.feat_dim)
        self.head = nn.Sequential(nn.Dropout(float(cfg.model.dropout)),
                                  nn.Linear(hidden, self.feat_dim))

        if bool(getattr(cfg.model, "ast_grad_checkpoint", False)):
            self.net.gradient_checkpointing_enable()

    @torch.no_grad()
    def _regrid_position_embeddings(self, n_frames_new: int):
        cfg = self.net.config
        emb = self.net.embeddings
        pe = emb.position_embeddings.data                  # [1, 2 + f*t, H]
        hidden = pe.shape[-1]
        f_dim = (int(cfg.num_mel_bins) - int(cfg.patch_size)) // int(cfg.frequency_stride) + 1
        t_old = (int(cfg.max_length) - int(cfg.patch_size)) // int(cfg.time_stride) + 1
        t_new = (int(n_frames_new) - int(cfg.patch_size)) // int(cfg.time_stride) + 1
        if t_new < 1:
            raise ValueError(
                f"clip too short for AST: {n_frames_new} frames gives t_dim={t_new}. "
                f"Increase audio.clip_seconds.")
        if t_new == t_old:
            return
        n_special = pe.shape[1] - f_dim * t_old
        cls = pe[:, :n_special, :]
        # patch order is (frequency-major, time-minor) — see ASTPatchEmbeddings
        grid = pe[:, n_special:, :].reshape(1, f_dim, t_old, hidden).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(f_dim, t_new), mode="bilinear", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, f_dim * t_new, hidden)
        emb.position_embeddings = nn.Parameter(torch.cat([cls, grid], dim=1))
        print(f"[AST] position grid re-gridded: t {t_old} -> {t_new} "
              f"(f={f_dim}, tokens {pe.shape[1]} -> {n_special + f_dim * t_new})")

    def forward(self, x):
        # x: [B, 1, F, T] log-mel  ->  AST wants [B, T, F]
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.transpose(1, 2).contiguous()
        T = x.shape[1]
        if T != self.n_frames:                    # tolerate off-by-one framing
            if T > self.n_frames:
                x = x[:, :self.n_frames, :]
            else:
                x = F.pad(x, (0, 0, 0, self.n_frames - T))
        out = self.net(input_values=x * self.in_scale)
        return self.head(out.pooler_output)


# ---------------------------------------------------------------------------
# timm (ImageNet-pretrained CNNs)
# ---------------------------------------------------------------------------
class TimmBackbone(nn.Module):
    """ImageNet-pretrained encoder on the 1-channel log-mel."""

    def __init__(self, cfg):
        super().__init__()
        import timm
        name = str(getattr(cfg.model, "timm_name", "efficientnet_b0"))
        self.net = timm.create_model(
            name, pretrained=bool(getattr(cfg.model, "timm_pretrained", True)),
            num_classes=0, in_chans=1, global_pool="avg")
        self.feat_dim = int(cfg.model.feat_dim)
        self.head = nn.Sequential(nn.Dropout(float(cfg.model.dropout)),
                                  nn.Linear(self.net.num_features, self.feat_dim))

    def forward(self, x):
        return self.head(self.net(x))


# ---------------------------------------------------------------------------
# Light CNN (capacity-controlled comparator)
# ---------------------------------------------------------------------------
def _blk(ci, co, pool=True):
    l = [nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True)]
    if pool:
        l.append(nn.MaxPool2d(2))
    return nn.Sequential(*l)


class CNNBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feat_dim = int(cfg.model.feat_dim)
        self.b1, self.b2 = _blk(1, 32), _blk(32, 64)
        self.b3, self.b4 = _blk(64, 128), _blk(128, 256, False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, self.feat_dim)
        self.drop = nn.Dropout(float(cfg.model.dropout))

    def forward(self, x):
        x = self.b4(self.b3(self.b2(self.b1(x))))
        return self.fc(self.drop(self.pool(x).flatten(1)))


def build_backbone(cfg):
    name = str(cfg.model.backbone).lower()
    if name == "ast":
        return ASTBackbone(cfg)
    if name in ("timm", "pretrained", "effnet"):
        return TimmBackbone(cfg)
    if name == "cnn":
        return CNNBackbone(cfg)
    raise ValueError(f"unknown backbone {name!r} (expected: ast | timm | cnn)")
