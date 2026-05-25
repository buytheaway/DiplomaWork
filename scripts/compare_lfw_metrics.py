"""Create a Markdown comparison table for custom and pretrained LFW metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def comparison_table(custom: dict[str, Any], pretrained: dict[str, Any]) -> str:
    rows = [custom, pretrained]
    lines = [
        "| Pipeline | Valid pairs | EER | EER threshold | Best accuracy | "
        "Best threshold | TAR@FAR=0.1 | TAR@FAR=0.01 | TAR@FAR=0.001 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        tar = row.get("tar_at_far", {})
        lines.append(
            "| "
            f"{row.get('pipeline_display_name', row.get('pipeline', 'unknown'))} | "
            f"{row.get('valid_pairs')} | "
            f"{_fmt(row.get('eer'))} | "
            f"{_fmt(row.get('eer_threshold'))} | "
            f"{_fmt(row.get('best_accuracy'))} | "
            f"{_fmt(row.get('best_accuracy_threshold'))} | "
            f"{_fmt(tar.get('0.1', {}).get('tar'))} | "
            f"{_fmt(tar.get('0.01', {}).get('tar'))} | "
            f"{_fmt(tar.get('0.001', {}).get('tar'))} |"
        )
    return "\n".join(lines)


def interpretation(custom: dict[str, Any], pretrained: dict[str, Any]) -> str:
    custom_eer = custom.get("eer")
    pretrained_eer = pretrained.get("eer")
    custom_acc = custom.get("best_accuracy")
    pretrained_acc = pretrained.get("best_accuracy")

    lines = [
        "## Interpretation",
        "",
        "- The custom row is the proposed custom Torch IR-50 pipeline.",
        "- The pretrained row is the ONNX/InsightFace baseline/reference.",
        "- Lower EER is better; higher best accuracy and TAR@FAR are better.",
        "- Do not claim the custom pipeline is better unless these metrics prove it.",
    ]
    if isinstance(custom_eer, int | float) and isinstance(pretrained_eer, int | float):
        if custom_eer < pretrained_eer:
            lines.append("- In this run, custom has lower EER than the pretrained baseline.")
        elif custom_eer > pretrained_eer:
            lines.append("- In this run, the pretrained baseline has lower EER than custom.")
        else:
            lines.append("- In this run, both pipelines have the same EER.")
    if isinstance(custom_acc, int | float) and isinstance(pretrained_acc, int | float):
        if custom_acc > pretrained_acc:
            lines.append("- In this run, custom has higher best-threshold accuracy.")
        elif custom_acc < pretrained_acc:
            lines.append("- In this run, the pretrained baseline has higher best-threshold accuracy.")
        else:
            lines.append("- In this run, both pipelines have the same best-threshold accuracy.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LFW metrics JSON files")
    parser.add_argument("--custom", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", default="reports/biometric_eval/lfw_comparison.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    custom = load_metrics(Path(args.custom))
    pretrained = load_metrics(Path(args.pretrained))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n".join(
        [
            "# LFW Verification Comparison",
            comparison_table(custom, pretrained),
            interpretation(custom, pretrained),
        ]
    )
    output.write_text(content + "\n", encoding="utf-8")
    print(f"Comparison saved to {output}")


if __name__ == "__main__":
    main()
