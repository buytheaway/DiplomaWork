import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark search latency and recall")
    parser.add_argument("--dataset", required=True, help="Dataset folder")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default="benchmarks")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images: list[tuple[Path, str]] = []
    for person_dir in dataset_dir.iterdir():
        if not person_dir.is_dir():
            continue
        label = person_dir.name
        for img in person_dir.glob("*.*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                images.append((img, label))

    if not images:
        raise SystemExit("No images found")

    random.shuffle(images)
    samples = images[: min(len(images), args.samples)]

    latencies = []
    found = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerow(["image", "label", "found", "latency_ms"])

        for image_path, label in samples:
            start = time.perf_counter()
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
            match = any(r.get("label") == label for r in results)
            if match:
                found += 1
            latencies.append(latency_ms)
            writer.writerow([str(image_path), label, int(match), f"{latency_ms:.3f}"])

    latencies_np = np.array(latencies, dtype=np.float32)
    summary = {
        "samples": len(latencies),
        "k": args.k,
        "recall_at_k": found / len(latencies),
        "latency_p50_ms": float(np.percentile(latencies_np, 50)),
        "latency_p95_ms": float(np.percentile(latencies_np, 95)),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
