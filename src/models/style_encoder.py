"""Lightweight style encoder for the disentanglement probe / optional FiLM.
Produces an L2-normalised style embedding from a mel patch."""
import torch, torch.nn as nn, torch.nn.functional as F


class StyleEncoder(nn.Module):
    def __init__(self, style_dim=128, proj_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, style_dim, 3, padding=1, bias=False), nn.BatchNorm2d(style_dim), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1))
        self.style_dim = style_dim
        self.proj = nn.Sequential(nn.Linear(style_dim, style_dim), nn.ReLU(True), nn.Linear(style_dim, proj_dim))

    def embed(self, mel): return self.enc(mel).flatten(1)
    def project(self, mel): return F.normalize(self.proj(self.embed(mel)), dim=-1)
