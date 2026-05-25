"""Run custom Torch IR-50 LFW preprocessing diagnostics.

This wrapper does not touch backend runtime, desktop, database, enroll/search,
or FAISS. It repeatedly calls ``evaluate_lfw_verification.py`` with small LFW
subsets and writes a markdown summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "evaluate_lfw_verification.py"


def _default_checkpoint() -> str:
    candidates = [
        ROOT / "handoff_lfw_eval" / "artifacts" / "best_lfw.pth",
        ROOT / "diplomcheckbackup" / "training" / "outputs_medium_lfw_finetune" / "best_lfw.pth",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def _default_detector() -> str:
    path = ROOT / "demo-model" / "models" / "det_10g.onnx"
    return str(path) if path.exists() else ""


def _configs(include_pretrained_alignment: bool) -> list[dict[str, str]]:
    configs = [
        {
            "name": "runtime_default",
            "custom_preprocess": "runtime",
            "color_order": "rgb",
            "normalization": "minus1_1",
            "detector": "custom",
        },
        {
            "name": "runtime_rgb",
            "custom_preprocess": "runtime",
            "color_order": "rgb",
            "normalization": "minus1_1",
            "detector": "custom",
        },
        {
            "name": "runtime_bgr",
            "custom_preprocess": "runtime",
            "color_order": "bgr",
            "normalization": "minus1_1",
            "detector": "custom",
        },
        {
            "name": "center_crop_rgb_minus1_1",
            "custom_preprocess": "center_crop",
            "color_order": "rgb",
            "normalization": "minus1_1",
            "detector": "none",
        },
        {
            "name": "center_crop_bgr_minus1_1",
            "custom_preprocess": "center_crop",
            "color_order": "bgr",
            "normalization": "minus1_1",
            "detector": "none",
        },
        {
            "name": "center_crop_rgb_imagenet",
            "custom_preprocess": "center_crop",
            "color_order": "rgb",
            "normalization": "imagenet",
            "detector": "none",
        },
        {
            "name": "center_crop_bgr_imagenet",
            "custom_preprocess": "center_crop",
            "color_order": "bgr",
            "normalization": "imagenet",
            "detector": "none",
        },
    ]
    if include_pretrained_alignment:
        configs.append(
            {
                "name": "pretrained_align_rgb_minus1_1",
                "custom_preprocess": "insightface_align",
                "color_order": "rgb",
                "normalization": "minus1_1",
                "detector": "pretrained",
            }
        )
    return configs


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
    lines = [
        "# Custom LFW Preprocessing Diagnostics",
        "",
        "| Config | Valid pairs | Skipped | EER | Best accuracy | Best threshold | "
        "TAR@FAR=0.01 | Mean positive | Mean negative |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {_format(row['valid_pairs'])} | "
            f"{_format(row['skipped_pairs'])} | {_format(row['eer'])} | "
            f"{_format(row['best_accuracy'])} | {_format(row['best_threshold'])} | "
            f"{_format(row['tar_far_0_01'])} | {_format(row['mean_positive_score'])} | "
            f"{_format(row['mean_negative_score'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `runtime_*` uses the current project custom extractor path.",
            "- `center_crop_*` bypasses detection and matches the LFW training/eval style more closely.",
            "- If LFW was used during fine-tuning, these results are training-stage/validation evidence, not an independent generalization test.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run custom LFW diagnostic ablations")
    parser.add_argument("--lfw-root", default="handoff_lfw_eval/lfw")
    parser.add_argument("--pairs-file", default="handoff_lfw_eval/lfw/pairs.txt")
    parser.add_argument("--checkpoint", default=_default_checkpoint())
    parser.add_argument("--output-dir", default="reports/biometric_eval/custom_diagnostics")
    parser.add_argument("--cache-dir", default="reports/biometric_eval/custom_diagnostics/cache")
    parser.add_argument("--max-pairs", type=int, default=600)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--onnx-detector-path", default=_default_detector())
    parser.add_argument("--include-pretrained-alignment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint:
        raise SystemExit("No custom checkpoint found. Pass --checkpoint.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TORCH_MODEL_PATH"] = str(Path(args.checkpoint))
    env["CUSTOM_BACKEND"] = "torch"
    env["CUSTOM_ALLOW_CENTER_CROP"] = "true"
    if args.onnx_detector_path:
        env["ONNX_DETECTOR_PATH"] = str(Path(args.onnx_detector_path))

    rows: list[dict[str, Any]] = []
    include_align = bool(args.include_pretrained_alignment and args.onnx_detector_path)
    for config in _configs(include_align):
        config_dir = output_dir / config["name"]
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
            args.checkpoint,
            "--custom-preprocess",
            config["custom_preprocess"],
            "--color-order",
            config["color_order"],
            "--normalization",
            config["normalization"],
            "--detector",
            config["detector"],
        ]
        print(f"Running {config['name']}...")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        metrics = json.loads((config_dir / "metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "config": config["name"],
                "valid_pairs": _metric(metrics, "valid_pairs"),
                "skipped_pairs": _metric(metrics, "skipped_pairs"),
                "eer": _metric(metrics, "eer"),
                "best_accuracy": _metric(metrics, "best_accuracy"),
                "best_threshold": _metric(metrics, "best_accuracy_threshold"),
                "tar_far_0_01": _metric(metrics, "tar_far_0_01"),
                "mean_positive_score": _metric(metrics, "mean_positive_score"),
                "mean_negative_score": _metric(metrics, "mean_negative_score"),
            }
        )

    _write_summary(output_dir, rows)
    print(f"Summary written to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
