"""Thin wrapper re-exporting the canonical IR-ResNet model from the training package.

The single source of truth lives in ``training/models/ir_resnet.py``.  This
module re-exports the symbols required by the backend so that the inference
and training models never silently diverge.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# Ensure the project root is on ``sys.path`` so that ``training.*`` is importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])  # .../DiplomaWork
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from training.models.ir_resnet import (  # noqa: E402
    LAYER_CONFIGS,
    IRBlock,
    ResNetIR,
)
from training.models.ir_resnet import (  # noqa: E402
    build_model as _training_build_model,
)

# ── Backend-specific config dataclass (kept for backward-compat) ─────────────


@dataclass
class ModelConfig:
    arch: str = "ir18"
    embedding_dim: int = 512
    norm_embeddings: bool = True


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-5),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class InsightFaceIResNet(nn.Module):
    """InsightFace-style IResNet backbone used by external ArcFace checkpoints."""

    def __init__(self, layers: list[int], embedding_dim: int = 512) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=0.0, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim)
        self.features = nn.BatchNorm1d(embedding_dim, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [IBasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.features(x)


def build_model(config: ModelConfig) -> nn.Module:
    """Build model using the *same* architecture as the training package."""
    if config.arch in {"iresnet50", "insightface_iresnet50"}:
        return InsightFaceIResNet([3, 4, 14, 3], embedding_dim=config.embedding_dim)
    if config.arch in {"iresnet100", "insightface_iresnet100"}:
        return InsightFaceIResNet([3, 13, 30, 3], embedding_dim=config.embedding_dim)
    return _training_build_model(config.arch, config.embedding_dim)


def forward_with_normalization(
    model: nn.Module, images: torch.Tensor, normalize: bool
) -> torch.Tensor:
    embeddings = model(images)
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


__all__ = [
    "ModelConfig",
    "build_model",
    "forward_with_normalization",
    "LAYER_CONFIGS",
    "IRBlock",
    "ResNetIR",
    "IBasicBlock",
    "InsightFaceIResNet",
]
