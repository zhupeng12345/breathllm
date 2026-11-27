"""TSD-Net-B architecture adapted for the UCDDB three-class benchmark.

Based on TSDNet.py from https://github.com/fdu-harry/Apnea-Interact-Xplainer
revision ef1544ffa04c5719fd4803d0591bd988b3b6899f.
Copyright (c) 2024 Shuaicong Hu. See ../AIX_LICENSE.txt.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        fan_out = module.kernel_size[0] * module.out_channels // module.groups
        nn.init.normal_(module.weight, std=math.sqrt(2 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class LocalAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.heads = 4
        group_width = width // self.heads
        self.values = nn.Linear(width, width)
        self.scales = nn.Linear(width, width)
        self.local = nn.ModuleList(
            nn.Conv1d(
                group_width,
                group_width,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=group_width,
            )
            for kernel in (3, 5, 7, 9)
        )
        self.mix = nn.Sequential(
            nn.Conv1d(width, 2 * width, 1, groups=group_width),
            nn.BatchNorm1d(2 * width),
            nn.GELU(),
            nn.Conv1d(2 * width, width, 1),
        )
        self.project = nn.Linear(width, width)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        batch, steps, width = signals.shape
        head_width = width // self.heads
        # Preserve the published inference code's layout and grouped fusion.
        scales = self.scales(signals).reshape(batch, self.heads, head_width, steps)
        local = [
            conv(scales[:, head]).reshape(batch, head_width, 1, steps)
            for head, conv in enumerate(self.local)
        ]
        fused = torch.cat(local, dim=2).reshape(batch, width, steps)
        return self.project(self.values(signals) * self.mix(fused).transpose(1, 2))


class GlobalAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.local = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.project = nn.Linear(width, width)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        batch, steps, width = signals.shape
        head_width = width // self.heads
        query = self.query(signals).reshape(batch, steps, self.heads, head_width)
        key, value = self.key_value(signals).reshape(
            batch, steps, 2, self.heads, head_width
        ).permute(2, 0, 3, 1, 4)
        local_values = value.transpose(1, 2).reshape(batch, steps, width).transpose(1, 2)
        attended = nn.functional.scaled_dot_product_attention(query.transpose(1, 2), key, value)
        attended = attended.transpose(1, 2).reshape(batch, steps, width)
        return self.project(attended + self.local(local_values).transpose(1, 2))


class TSDBlock(nn.Module):
    def __init__(self, width: int, mlp_ratio: int, global_heads: int = 0) -> None:
        super().__init__()
        mlp_width = width * mlp_ratio
        self.norm_attention = nn.LayerNorm(width, eps=1e-6)
        self.attention = (
            GlobalAttention(width, global_heads) if global_heads else LocalAttention(width)
        )
        self.norm_mlp = nn.LayerNorm(width, eps=1e-6)
        self.expand = nn.Linear(width, mlp_width)
        self.temporal = nn.Conv1d(mlp_width, mlp_width, 3, padding=1, groups=mlp_width)
        self.activation = nn.GELU()
        self.contract = nn.Linear(mlp_width, width)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        signals = signals + self.attention(self.norm_attention(signals))
        residual = self.expand(self.norm_mlp(signals))
        local = self.temporal(residual.transpose(1, 2)).transpose(1, 2)
        return signals + self.contract(self.activation(residual + local))


class TSDNet3C(nn.Module):
    """TSD-Net-B with eight pooled features and a three-class logit head."""

    def __init__(self, channel_index: int) -> None:
        super().__init__()
        self.channel_index = channel_index
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 2, stride=2),
        )
        widths = (64, 128, 256, 512)
        self.stages = nn.ModuleList(
            nn.Sequential(*(TSDBlock(width, ratio, heads) for heads in head_counts))
            for width, ratio, head_counts in zip(
                widths,
                (4, 4, 4, 2),
                ((0, 0), (0, 0), (0, 8, 0, 8, 0, 8, 0, 8), (16,)),
            )
        )
        self.patch_embeddings = nn.ModuleList(
            nn.Conv1d(width_in, width_out, 1, stride=2)
            for width_in, width_out in zip(widths, widths[1:])
        )
        self.patch_norms = nn.ModuleList(
            nn.LayerNorm(width, eps=1e-5) for width in widths[1:]
        )
        self.classifier = nn.Linear(8, 3)
        self.apply(_init_weights)

    def forward_features(self, signals: torch.Tensor) -> torch.Tensor:
        # Use the selected channel and the released per-window z-score.
        channel = signals[:, self.channel_index:self.channel_index + 1]
        if channel.shape[-1] != 256:
            raise ValueError(f"TSD-Net expects 256 contextual samples, got {channel.shape[-1]}")
        channel = channel - channel.mean(dim=-1, keepdim=True)
        channel = channel / (channel.std(dim=-1, keepdim=True, correction=0) + 1e-8)
        features = self.stages[0](self.stem(channel).transpose(1, 2))
        for embed, norm, stage in zip(self.patch_embeddings, self.patch_norms, self.stages[1:]):
            features = stage(norm(embed(features.transpose(1, 2)).transpose(1, 2)))
        return features.mean(dim=-1)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(signals))


def build_aix_model(name: str, channel_names: list[str]) -> TSDNet3C:
    """Build the single-modality AIX baseline using airflow."""
    if name != "AIX-TSDNet-3C":
        raise ValueError(f"Unsupported single-modality AIX method {name}")
    return TSDNet3C(channel_names.index("airflow"))


def aix_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    cross_entropy = nn.functional.cross_entropy(logits, targets, reduction="none")
    focus = (1 - torch.exp(-cross_entropy)).square()
    weights = class_weights[targets]
    return (weights * focus * cross_entropy).sum() / weights.sum()
