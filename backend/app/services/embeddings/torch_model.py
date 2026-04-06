"""Thin wrapper re-exporting the canonical IR-ResNet model from the training package.

The single source of truth lives in ``training/models/ir_resnet.py``.  This
module re-exports the symbols required by the backend so that the inference
and training models never silently diverge.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

# Ensure the project root is on ``sys.path`` so that ``training.*`` is importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[5])  # …/DiplomaWork
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from training.models.ir_resnet import (  # noqa: E402
    LAYER_CONFIGS,
    IRBlock,
    ResNetIR,
    build_model as _training_build_model,
)


# ── Backend-specific config dataclass (kept for backward-compat) ─────────────


@dataclass
class ModelConfig:
    arch: str = "ir18"
    embedding_dim: int = 512
    norm_embeddings: bool = True


def build_model(config: ModelConfig) -> nn.Module:
    """Build model using the *same* architecture as the training package."""
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
]
