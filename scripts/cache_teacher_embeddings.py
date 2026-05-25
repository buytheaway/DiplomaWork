"""Cache pretrained teacher embeddings for identity-folder training images.

This script is standalone and does not call the backend API, database, desktop
UI, enroll/search flow, runtime checkpoint, or FAISS index.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
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
os.environ.setdefault("PRETRAINED_BACKEND", "onnx")
os.environ.setdefault("EMBEDDING_BACKEND", "onnx")
os.environ.setdefault("DETECTION_BACKEND", "none")
os.environ.setdefault("ONNX_DETECTOR_PATH", "models/det_10g.onnx")
os.environ.setdefault("ONNX_EMBEDDER_PATH", "models/w600k_r50.onnx")

from app.services.embeddings.interface import (  # noqa: E402
    FaceProcessingError,
    NoFaceDetectedError,
)
from app.services.embeddings.onnx_extractor import (  # noqa: E402
    _ArcFaceEmbedder,
    _ensure_cv2,
    _ensure_ort,
    _preferred_ort_providers,
)
from scripts.evaluate_lfw_verification import build_extractor, normalize_embedding  # noqa: E402
from training.datasets.folder_dataset import IMAGE_EXTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache teacher embeddings for distillation")
    parser.add_argument("--image-root", default="datasets/celeba_faces/train")
    parser.add_argument("--teacher-pipeline", choices=["pretrained"], default="pretrained")
    parser.add_argument(
        "--output",
        default="reports/teacher_cache/celeba_train_teacher_embeddings",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--input-aligned",
        action="store_true",
        help="Treat input images as already aligned 112x112 crops and skip detection.",
    )
    parser.add_argument("--embedder-path", default="models/w600k_r50.onnx")
    return parser.parse_args()


def _set_pretrained_defaults() -> None:
    os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    os.environ.setdefault("DEFAULT_PIPELINE", "pretrained")
    os.environ.setdefault("ENABLE_PRETRAINED_PIPELINE", "true")
    os.environ.setdefault("ENABLE_CUSTOM_PIPELINE", "false")
    os.environ.setdefault("PRETRAINED_BACKEND", "onnx")
    os.environ.setdefault("EMBEDDING_BACKEND", "onnx")
    os.environ.setdefault("DETECTION_BACKEND", "none")
    os.environ.setdefault("ONNX_DETECTOR_PATH", "models/det_10g.onnx")
    os.environ.setdefault("ONNX_EMBEDDER_PATH", "models/w600k_r50.onnx")


def iter_images(root: Path, max_images: int | None) -> list[Path]:
    images: list[Path] = []
    for identity_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        for image_path in sorted(identity_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            images.append(image_path)
            if max_images is not None and len(images) >= max_images:
                return images
    return images


def embedding_relative_path(root: Path, output: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(root).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16]
    identity = image_path.parent.name
    return Path("embeddings") / identity / f"{image_path.stem}-{digest}.npy"


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["image_path", "identity", "embedding_path", "status", "error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_aligned_embedder(embedder_path: Path) -> _ArcFaceEmbedder:
    if not embedder_path.is_absolute():
        embedder_path = ROOT / embedder_path
    if not embedder_path.exists():
        raise FileNotFoundError(f"ONNX embedder model not found: {embedder_path}")
    ort = _ensure_ort()
    providers = _preferred_ort_providers(ort)
    session = ort.InferenceSession(str(embedder_path), providers=providers)
    return _ArcFaceEmbedder(session)


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    output = Path(args.output)
    if not image_root.exists():
        raise SystemExit(f"Image root not found: {image_root}")
    if args.max_images is not None and args.max_images <= 0:
        raise SystemExit("--max-images must be positive")

    _set_pretrained_defaults()
    extractor = None
    aligned_embedder = None
    cv2 = None
    if args.input_aligned:
        aligned_embedder = build_aligned_embedder(Path(args.embedder_path))
        cv2 = _ensure_cv2()
    else:
        extractor, _settings = build_extractor(args.teacher_pipeline, args.device)
    image_paths = iter_images(image_root, args.max_images)
    if not image_paths:
        raise SystemExit(f"No images found under {image_root}")

    rows: list[dict[str, Any]] = []
    ok_count = 0
    skipped_count = 0
    start = time.time()
    for image_path in tqdm(image_paths, desc="Teacher cache"):
        relative_image = image_path.relative_to(image_root).as_posix()
        relative_embedding = embedding_relative_path(image_root, output, image_path)
        embedding_path = output / relative_embedding
        row = {
            "image_path": relative_image,
            "identity": image_path.parent.name,
            "embedding_path": relative_embedding.as_posix(),
            "status": "ok",
            "error": "",
        }
        if args.resume and embedding_path.exists():
            rows.append(row)
            ok_count += 1
            continue

        try:
            if aligned_embedder is not None:
                assert cv2 is not None
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Cannot decode image")
                embedding = normalize_embedding(aligned_embedder(image))
            else:
                assert extractor is not None
                face = extractor.extract_largest_embedding(image_path.read_bytes())
                embedding = normalize_embedding(face.embedding)
            embedding_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embedding_path, embedding.astype(np.float32))
            ok_count += 1
        except (NoFaceDetectedError, FaceProcessingError) as exc:
            row["status"] = "skipped"
            row["error"] = str(exc)
            skipped_count += 1
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            skipped_count += 1
        rows.append(row)

    write_metadata(output / "metadata.csv", rows)
    summary = {
        "image_root": str(image_root),
        "output": str(output),
        "total_images": len(image_paths),
        "cached_embeddings": ok_count,
        "skipped_or_error": skipped_count,
        "elapsed_s": round(time.time() - start, 2),
        "teacher_pipeline": args.teacher_pipeline,
        "input_aligned": bool(args.input_aligned),
    }
    (output / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
