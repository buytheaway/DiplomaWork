import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CONFIG = """
DATASET FORMAT:
  dataset/<identity>/*.(jpg|jpeg|png|bmp)

FACE POLICY:
  strict single-face only (skip 0 faces or >1 faces)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark search latency and recall")
    parser.add_argument("--dataset", required=True, help="Dataset folder")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default="benchmarks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images: list[tuple[Path, str]] = []
    for person_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        label = person_dir.name
        for img in sorted(
            (p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
            key=lambda p: p.name,
        ):
            images.append((img, label))

    if not images:
        raise SystemExit("No images found")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    samples = images[: min(len(images), args.samples)]

    latencies: list[float] = []
    found = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"

    total_start = time.perf_counter()
    with csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerow(["image", "label", "ok", "latency_ms", "top1_label", "topk_labels"])

        for image_path, label in samples:
            start = time.perf_counter()
            ok = 1
            top1_label = ""
            topk_labels: list[str] = []
            try:
                with image_path.open("rb") as handle:
                    response = requests.post(
                        f"{args.base_url.rstrip('/')}/v1/search",
                        params={"k": args.k},
                        files={"file": handle},
                        timeout=30,
                    )
                latency_ms = (time.perf_counter() - start) * 1000
                response.raise_for_status()
                results = response.json().get("results", [])
                topk_labels = [r.get("label") or "" for r in results]
                top1_label = topk_labels[0] if topk_labels else ""
                match = any(lab == label for lab in topk_labels)
                if match:
                    found += 1
            except Exception:
                ok = 0
                latency_ms = (time.perf_counter() - start) * 1000
                match = False

            latencies.append(latency_ms)
            writer.writerow(
                [
                    str(image_path),
                    label,
                    ok,
                    f"{latency_ms:.3f}",
                    top1_label,
                    "|".join(topk_labels),
                ]
            )

    total_time = time.perf_counter() - total_start
    latencies_np = np.array(latencies, dtype=np.float32)
    summary = {
        "samples": len(latencies),
        "k": args.k,
        "recall_at_k": found / len(latencies),
        "latency_p50_ms": float(np.percentile(latencies_np, 50)),
        "latency_p95_ms": float(np.percentile(latencies_np, 95)),
        "latency_p99_ms": float(np.percentile(latencies_np, 99)),
        "latency_avg_ms": float(np.mean(latencies_np)),
        "qps": float(len(latencies) / total_time) if total_time > 0 else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
