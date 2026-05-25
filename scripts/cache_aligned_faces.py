"""Cache SCRFD/ArcFace-aligned face crops for custom training experiments.

This script is intentionally offline and isolated. It does not call the
backend API, does not touch runtime checkpoints, and does not write FAISS
indexes or database rows. Outputs should live under ignored experiment paths
such as ``training/runs/aligned_faces``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("DEFAULT_PIPELINE", "pretrained")
os.environ.setdefault("ENABLE_PRETRAINED_PIPELINE", "true")
os.environ.setdefault("ENABLE_CUSTOM_PIPELINE", "false")
os.environ.setdefault("ONNX_DETECTOR_PATH", "models/det_10g.onnx")
os.environ.setdefault("ONNX_EMBEDDER_PATH", "models/w600k_r50.onnx")
os.environ.setdefault("PRETRAINED_BACKEND", "onnx")

from app.services.embeddings.onnx_extractor import (  # noqa: E402
    _align_face,
    _ensure_cv2,
    _ensure_ort,
    _preferred_ort_providers,
    _SCRFDDetector,
)
from training.datasets.folder_dataset import IMAGE_EXTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache detector-aligned 112x112 face crops for training",
    )
    parser.add_argument("--image-root", default="datasets/celeba_faces/train")
    parser.add_argument(
        "--output-root",
        default="training/runs/aligned_faces/celeba_train_scrfd_112",
    )
    parser.add_argument("--detector-path", default="models/det_10g.onnx")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-size", type=int, default=112)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def iter_images(root: Path, max_images: int | None) -> list[Path]:
    paths: list[Path] = []
    for identity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for image_path in sorted(identity_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            paths.append(image_path)
            if max_images is not None and len(paths) >= max_images:
                return paths
    return paths


def output_path_for(root: Path, output_root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(root)
    return output_root / relative.parent / f"{relative.stem}.jpg"


def build_detector(detector_path: Path) -> _SCRFDDetector:
    if not detector_path.is_absolute():
        detector_path = ROOT / detector_path
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector model not found: {detector_path}")

    ort = _ensure_ort()
    providers = _preferred_ort_providers(ort)
    session = ort.InferenceSession(str(detector_path), providers=providers)
    return _SCRFDDetector(session)


def select_face(boxes: np.ndarray, scores: np.ndarray) -> int:
    """Pick the largest detected face, using score only as a tie-breaker."""
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    areas = widths * heights
    ranking = areas + (scores.astype(np.float32) * 1e-3)
    return int(np.argmax(ranking))


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "source_path",
        "identity",
        "aligned_path",
        "status",
        "error",
        "detection_score",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    detector_path = Path(args.detector_path)

    if not image_root.exists():
        raise SystemExit(f"Image root not found: {image_root}")
    if args.max_images is not None and args.max_images <= 0:
        raise SystemExit("--max-images must be positive")
    if args.output_size <= 0:
        raise SystemExit("--output-size must be positive")
    if not (1 <= args.jpeg_quality <= 100):
        raise SystemExit("--jpeg-quality must be between 1 and 100")

    cv2 = _ensure_cv2()
    detector = build_detector(detector_path)
    image_paths = iter_images(image_root, args.max_images)
    if not image_paths:
        raise SystemExit(f"No images found under {image_root}")

    rows: list[dict[str, Any]] = []
    ok_count = 0
    skipped_count = 0
    started = time.time()

    for image_path in tqdm(image_paths, desc="Align faces"):
        relative_source = image_path.relative_to(image_root).as_posix()
        aligned_path = output_path_for(image_root, output_root, image_path)
        relative_aligned = aligned_path.relative_to(output_root).as_posix()
        row: dict[str, Any] = {
            "source_path": relative_source,
            "identity": image_path.parent.name,
            "aligned_path": relative_aligned,
            "status": "ok",
            "error": "",
            "detection_score": "",
            "bbox_x1": "",
            "bbox_y1": "",
            "bbox_x2": "",
            "bbox_y2": "",
        }

        if args.resume and aligned_path.exists():
            rows.append(row)
            ok_count += 1
            continue

        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Cannot decode image")
            boxes, keypoints, scores = detector(
                image,
                score_thresh=args.score_threshold,
                iou_thresh=args.iou_threshold,
            )
            if boxes.size == 0 or keypoints is None:
                raise ValueError("No face landmarks detected")

            index = select_face(boxes, scores)
            aligned = _align_face(image, keypoints[index], args.output_size)
            aligned_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(
                str(aligned_path),
                aligned,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
            )
            if not ok:
                raise ValueError("Failed to write aligned crop")

            box = boxes[index]
            row.update(
                {
                    "detection_score": round(float(scores[index]), 6),
                    "bbox_x1": round(float(box[0]), 3),
                    "bbox_y1": round(float(box[1]), 3),
                    "bbox_x2": round(float(box[2]), 3),
                    "bbox_y2": round(float(box[3]), 3),
                }
            )
            ok_count += 1
        except Exception as exc:
            row["status"] = "skipped"
            row["error"] = str(exc)
            skipped_count += 1
        rows.append(row)

    write_metadata(output_root / "metadata.csv", rows)
    summary = {
        "image_root": str(image_root),
        "output_root": str(output_root),
        "detector_path": str(detector_path),
        "total_images": len(image_paths),
        "aligned_images": ok_count,
        "skipped_or_error": skipped_count,
        "output_size": args.output_size,
        "score_threshold": args.score_threshold,
        "elapsed_s": round(time.time() - started, 2),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
