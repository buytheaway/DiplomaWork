from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    *,
    head_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
    scaler_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_dir = ensure_dir(output_dir)
    ckpt = {
        "epoch": epoch,
        "state_dict": model_state,
        "optimizer": optimizer_state,
    }
    if head_state is not None:
        ckpt["head_state_dict"] = head_state
    if scheduler_state is not None:
        ckpt["scheduler"] = scheduler_state
    if scaler_state is not None:
        ckpt["scaler"] = scaler_state
    if metadata is not None:
        ckpt["metadata"] = metadata
    path = output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
    torch.save(ckpt, path)
    return path


def save_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    output_dir = ensure_dir(output_dir)
    path = output_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
