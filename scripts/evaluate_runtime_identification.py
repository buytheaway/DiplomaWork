"""Evaluate end-to-end runtime identification through the backend search API.

This is intentionally separate from LFW biometric verification. It measures the
deployed API workflow on labeled image folders:

    label folder -> backend /v1/search -> predicted label

It should be used with a held-out folder that was not used for enrollment when
reporting formal desktop/runtime identification accuracy.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class EvalItem:
    expected_label: str
    image_path: Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def discover_items(folder: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    for label_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
        label = label_dir.name.strip()
        if not label:
            continue
        for image_path in sorted(label_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                items.append(EvalItem(label, image_path.resolve()))
    return items


def request_search(
    *,
    base_url: str,
    api_key: str,
    item: EvalItem,
    pipeline: str,
    top_k: int,
    source: str,
    multi_face: bool,
    timeout: int,
) -> dict[str, Any]:
    mime = mimetypes.guess_type(str(item.image_path))[0] or "image/jpeg"
    with item.image_path.open("rb") as handle:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/search",
            params={
                "pipeline": pipeline,
                "k": top_k,
                "source": source,
                "multi_face": str(multi_face).lower(),
            },
            headers={"X-API-Key": api_key},
            files={"file": (item.image_path.name, handle, mime)},
            timeout=timeout,
        )
    payload: dict[str, Any]
    try:
        payload = response.json()
    except Exception:
        payload = {"detail": response.text[:500]}
    payload["_status_code"] = response.status_code
    return payload


def classify(expected: str, payload: dict[str, Any]) -> tuple[str, str | None, float | None]:
    if payload.get("_status_code") != 200:
        return "error", None, None
    results = payload.get("results") or []
    top = results[0] if results else {}
    predicted = top.get("label")
    score = top.get("score")
    if payload.get("decision") != "match" or not predicted:
        return "no_match", predicted, score
    if predicted == expected:
        return "correct", predicted, score
    return "false_match", predicted, score


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Runtime Identification Evaluation",
        "",
        "This report measures end-to-end backend identification behavior. It is not",
        "LFW pair verification and must not be mixed with FAR/FRR/EER results.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "items",
        "correct",
        "false_match",
        "no_match",
        "errors",
        "accuracy",
        "false_match_rate",
        "no_match_rate",
        "mean_latency_ms",
        "p95_latency_ms",
    ):
        lines.append(f"| {key} | {summary.get(key)} |")
    lines.extend(
        [
            "",
            "## Per Label",
            "",
            "| Label | Items | Correct | False match | No match | Errors | Accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["per_label"]:
        lines.append(
            "| {label} | {items} | {correct} | {false_match} | {no_match} | {errors} | {accuracy} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Use a held-out folder not used during enrollment for a formal desktop/runtime accuracy claim.",
            "- If this report is run on the enrollment folder, it is only a smoke test.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return round(values[index], 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate runtime identification accuracy via backend /v1/search."
    )
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output-dir", default="reports/runtime_identification_eval")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--pipeline", choices=["custom", "pretrained"], default="custom")
    parser.add_argument("--top-k", type=positive_int, default=3)
    parser.add_argument("--source", choices=["manual", "webcam"], default="manual")
    parser.add_argument("--multi-face", action="store_true")
    parser.add_argument("--timeout", type=positive_int, default=120)
    parser.add_argument("--max-images", type=positive_int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = resolve_path(args.folder)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = load_env(ROOT / ".env")
    api_key = env.get("API_KEY", "")
    items = discover_items(folder)
    if args.max_images:
        items = items[: args.max_images]
    if not items:
        raise SystemExit(f"No images found under {folder}")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    by_label: dict[str, Counter[str]] = {}
    latencies: list[float] = []
    for item in items:
        payload = request_search(
            base_url=args.base_url,
            api_key=api_key,
            item=item,
            pipeline=args.pipeline,
            top_k=args.top_k,
            source=args.source,
            multi_face=args.multi_face,
            timeout=args.timeout,
        )
        outcome, predicted, score = classify(item.expected_label, payload)
        latency = payload.get("latency_ms")
        if isinstance(latency, int | float):
            latencies.append(float(latency))
        counts[outcome] += 1
        by_label.setdefault(item.expected_label, Counter())[outcome] += 1
        rows.append(
            {
                "expected_label": item.expected_label,
                "predicted_label": predicted,
                "score": score,
                "outcome": outcome,
                "latency_ms": latency,
                "status_code": payload.get("_status_code"),
                "decision": payload.get("decision"),
                "best_score": payload.get("best_score"),
                "search_mode": payload.get("search_mode"),
                "candidate_k": payload.get("candidate_k"),
                "image_path": str(item.image_path),
            }
        )

    total = len(items)
    correct = counts["correct"]
    false_match = counts["false_match"]
    no_match = counts["no_match"]
    errors = counts["error"]
    per_label = []
    for label in sorted(by_label):
        c = by_label[label]
        label_total = sum(c.values())
        per_label.append(
            {
                "label": label,
                "items": label_total,
                "correct": c["correct"],
                "false_match": c["false_match"],
                "no_match": c["no_match"],
                "errors": c["error"],
                "accuracy": round(c["correct"] / label_total, 6) if label_total else 0.0,
            }
        )
    summary = {
        "folder": str(folder),
        "pipeline": args.pipeline,
        "source": args.source,
        "multi_face": args.multi_face,
        "top_k": args.top_k,
        "items": total,
        "correct": correct,
        "false_match": false_match,
        "no_match": no_match,
        "errors": errors,
        "accuracy": round(correct / total, 6),
        "false_match_rate": round(false_match / total, 6),
        "no_match_rate": round(no_match / total, 6),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "per_label": per_label,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(output_dir / "summary.md", summary)
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
