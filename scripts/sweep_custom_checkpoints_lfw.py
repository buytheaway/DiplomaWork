"""Sweep custom Torch checkpoints on a small LFW verification subset."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "evaluate_lfw_verification.py"


def _default_checkpoint_dir() -> str:
    candidates = [
        ROOT / "diplomcheckbackup" / "training" / "outputs_medium_lfw_finetune",
        ROOT / "training" / "outputs_medium_lfw_finetune",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"epoch_(\d+)", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def _find_checkpoints(directory: Path, include_best: bool) -> list[Path]:
    checkpoints = sorted(directory.glob("checkpoint_epoch_*.pth"), key=_checkpoint_sort_key)
    if include_best:
        best = directory / "best_lfw.pth"
        if best.exists():
            checkpoints.append(best)
    return checkpoints


def _metric(metrics: dict[str, Any], key: str) -> Any:
    if key == "tar_far_0_01":
        return metrics.get("tar_at_far", {}).get("0.01", {}).get("tar")
    return metrics.get(key)


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    best_eer = min(rows, key=lambda row: row["eer"])
    best_tar = max(rows, key=lambda row: row["tar_far_0_01"] or -1.0)
    best_acc = max(rows, key=lambda row: row["best_accuracy"])
    lines = [
        "# Custom Checkpoint LFW Sweep",
        "",
        "| Checkpoint | Valid pairs | Skipped | EER | Best accuracy | Best threshold | TAR@FAR=0.01 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {_format(row['valid_pairs'])} | "
            f"{_format(row['skipped_pairs'])} | {_format(row['eer'])} | "
            f"{_format(row['best_accuracy'])} | {_format(row['best_threshold'])} | "
            f"{_format(row['tar_far_0_01'])} |"
        )
    lines.extend(
        [
            "",
            "## Best by Criterion",
            "",
            f"- Lowest EER: `{best_eer['checkpoint']}` = `{_format(best_eer['eer'])}`",
            f"- Highest TAR@FAR=0.01: `{best_tar['checkpoint']}` = `{_format(best_tar['tar_far_0_01'])}`",
            f"- Highest best accuracy: `{best_acc['checkpoint']}` = `{_format(best_acc['best_accuracy'])}`",
            "",
            "## Data Leakage Note",
            "",
            "If these checkpoints were selected using LFW during training/fine-tuning, the sweep is training-stage/validation evidence, not an independent generalization test.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep custom checkpoints on LFW")
    parser.add_argument("--lfw-root", default="handoff_lfw_eval/lfw")
    parser.add_argument("--pairs-file", default="handoff_lfw_eval/lfw/pairs.txt")
    parser.add_argument("--checkpoint-dir", default=_default_checkpoint_dir())
    parser.add_argument("--output-dir", default="reports/biometric_eval/custom_checkpoint_sweep")
    parser.add_argument("--cache-dir", default="reports/biometric_eval/custom_checkpoint_sweep/cache")
    parser.add_argument("--max-pairs", type=int, default=600)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--custom-preprocess", default="center_crop")
    parser.add_argument("--color-order", default="rgb")
    parser.add_argument("--normalization", default="minus1_1")
    parser.add_argument("--include-best", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise SystemExit(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoints = _find_checkpoints(checkpoint_dir, args.include_best)
    if not checkpoints:
        raise SystemExit(f"No checkpoint_epoch_*.pth files found in {checkpoint_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["CUSTOM_BACKEND"] = "torch"
    env["CUSTOM_ALLOW_CENTER_CROP"] = "true"
    for checkpoint in checkpoints:
        config_name = checkpoint.stem
        config_dir = output_dir / config_name
        env["TORCH_MODEL_PATH"] = str(checkpoint)
        command = [
            sys.executable,
            str(EVALUATOR),
            "--lfw-root",
            args.lfw_root,
            "--pairs-file",
            args.pairs_file,
            "--pipeline",
            "custom",
            "--output-dir",
            str(config_dir),
            "--cache-dir",
            args.cache_dir,
            "--max-pairs",
            str(args.max_pairs),
            "--device",
            args.device,
            "--checkpoint",
            str(checkpoint),
            "--custom-preprocess",
            args.custom_preprocess,
            "--color-order",
            args.color_order,
            "--normalization",
            args.normalization,
            "--detector",
            "none",
        ]
        print(f"Running {checkpoint.name}...")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        metrics = json.loads((config_dir / "metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "checkpoint": checkpoint.name,
                "valid_pairs": _metric(metrics, "valid_pairs"),
                "skipped_pairs": _metric(metrics, "skipped_pairs"),
                "eer": _metric(metrics, "eer"),
                "best_accuracy": _metric(metrics, "best_accuracy"),
                "best_threshold": _metric(metrics, "best_accuracy_threshold"),
                "tar_far_0_01": _metric(metrics, "tar_far_0_01"),
            }
        )

    _write_summary(output_dir, rows)
    print(f"Summary written to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
