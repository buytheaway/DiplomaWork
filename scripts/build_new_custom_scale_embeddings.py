"""Build a separate large-scale FAISS index with the new custom Torch model.

The script writes embeddings, mapping and FAISS files under an explicit output
directory. It does not write production database rows and does not touch
backend/data/index unless the caller explicitly points --output-dir there.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings, settings  # noqa: E402
from app.services.embeddings.interface import (  # noqa: E402
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    create_extractor,
)

EXPECTED_MODEL_NAME = "torch_insightface_iresnet100"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ScaleConfig:
    sources: list[Path]
    model_name: str
    output_dir: Path
    max_images: int
    batch_size: int
    index_type: str
    nlist: int
    m: int
    nbits: int
    nprobe: int
    dry_run: bool


@dataclass
class ExtractSummary:
    images_seen: int = 0
    embeddings_created: int = 0
    skipped_no_face: int = 0
    skipped_invalid: int = 0
    skipped_multiple_faces: int = 0
    errors: int = 0
    extract_time_s: float = 0.0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def batch_size_value(value: str) -> int:
    if value.strip().lower() == "auto":
        return 4096
    return positive_int(value)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def custom_runtime_settings(base: Settings) -> Settings:
    updates = {
        "embedding_backend": base.custom_backend,
        "index_path": base.custom_index_path,
        "detection_backend": base.custom_detection_backend,
        "allow_center_crop": base.custom_allow_center_crop,
    }
    if base.custom_min_det_score is not None:
        updates["min_det_score"] = base.custom_min_det_score
    if base.custom_face_crop_margin is not None:
        updates["face_crop_margin"] = base.custom_face_crop_margin
    if base.custom_yolo_imgsz is not None:
        updates["yolo_imgsz"] = base.custom_yolo_imgsz
    return base.model_copy(update=updates)


def validate_runtime(custom_settings: Settings, expected_model_name: str) -> None:
    if expected_model_name != EXPECTED_MODEL_NAME:
        raise RuntimeError(f"--model-name must be {EXPECTED_MODEL_NAME}")
    if custom_settings.embedding_backend != "torch":
        raise RuntimeError("CUSTOM_BACKEND must resolve to torch")
    if custom_settings.torch_model_arch != "insightface_iresnet100":
        raise RuntimeError("TORCH_MODEL_ARCH must be insightface_iresnet100")
    if custom_settings.torch_preprocess != "runtime_fallback_center_crop":
        raise RuntimeError("TORCH_PREPROCESS must be runtime_fallback_center_crop")
    if custom_settings.torch_tta != "hflip":
        raise RuntimeError("TORCH_TTA must be hflip")


def discover_images(sources: list[Path], max_images: int) -> list[Path]:
    images: list[Path] = []
    for source in sources:
        if source.is_file() and source.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(source.resolve())
        elif source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    images.append(path.resolve())
                    if len(images) >= max_images:
                        return images
        if len(images) >= max_images:
            return images
    return images[:max_images]


def infer_identity(path: Path, sources: list[Path]) -> str:
    for source in sources:
        if source.is_dir():
            try:
                relative = path.relative_to(source)
            except ValueError:
                continue
            return relative.parts[0] if len(relative.parts) > 1 else source.name
    return path.parent.name


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_embeddings(
    image_paths: list[Path],
    *,
    config: ScaleConfig,
    custom_settings: Settings,
) -> tuple[Path, Path, ExtractSummary]:
    validate_runtime(custom_settings, config.model_name)
    extractor = create_extractor(custom_settings)
    if extractor.model_name != config.model_name:
        raise RuntimeError(f"Extractor model mismatch: {extractor.model_name}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = config.output_dir / "embeddings.float32.memmap"
    mapping_path = config.output_dir / "embedding_mapping.csv"
    vectors = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(len(image_paths), 512))

    summary = ExtractSummary(images_seen=len(image_paths))
    started = time.perf_counter()
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "image_path", "identity", "model_name"],
        )
        writer.writeheader()
        row_id = 0
        for image_path in image_paths:
            try:
                vector = extractor.extract_embedding(image_path.read_bytes())
            except NoFaceDetectedError:
                summary.skipped_no_face += 1
                continue
            except MultipleFacesDetectedError:
                summary.skipped_multiple_faces += 1
                continue
            except InvalidImageError:
                summary.skipped_invalid += 1
                continue
            except Exception:  # noqa: BLE001
                summary.errors += 1
                continue

            if vector.shape[0] != 512 or not np.isfinite(vector).all():
                summary.errors += 1
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0:
                summary.errors += 1
                continue
            vectors[row_id] = (vector / norm).astype(np.float32)
            writer.writerow(
                {
                    "row_id": row_id,
                    "image_path": str(image_path),
                    "identity": infer_identity(image_path, config.sources),
                    "model_name": config.model_name,
                }
            )
            row_id += 1
            summary.embeddings_created += 1
            if row_id % max(config.batch_size, 1) == 0:
                vectors.flush()

    vectors.flush()
    summary.extract_time_s = time.perf_counter() - started
    del vectors
    return memmap_path, mapping_path, summary


def build_index(
    memmap_path: Path,
    *,
    count: int,
    config: ScaleConfig,
) -> dict[str, Any]:
    if count <= 0:
        raise RuntimeError("No embeddings were created; cannot build FAISS index")
    vectors = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(count, 512))
    started = time.perf_counter()
    train_time_s = 0.0

    if config.index_type == "flat":
        index: faiss.Index = faiss.IndexIDMap2(faiss.IndexFlatIP(512))
        params: dict[str, Any] = {}
    elif config.index_type == "ivfpq":
        if 512 % config.m != 0:
            raise ValueError("512 must be divisible by --m")
        train_size = min(count, max(config.nlist, config.batch_size))
        train_vectors = np.ascontiguousarray(vectors[:train_size], dtype=np.float32)
        effective_nlist = min(config.nlist, train_size)
        quantizer = faiss.IndexFlatIP(512)
        base = faiss.IndexIVFPQ(
            quantizer,
            512,
            effective_nlist,
            config.m,
            config.nbits,
            faiss.METRIC_INNER_PRODUCT,
        )
        train_started = time.perf_counter()
        base.train(train_vectors)
        train_time_s = time.perf_counter() - train_started
        base.nprobe = min(config.nprobe, effective_nlist)
        index = faiss.IndexIDMap2(base)
        params = {
            "nlist": effective_nlist,
            "requested_nlist": config.nlist,
            "m": config.m,
            "nbits": config.nbits,
            "nprobe": int(base.nprobe),
            "train_size": train_size,
        }
    else:
        raise ValueError("--index-type must be flat or ivfpq")

    add_started = time.perf_counter()
    offset = 0
    while offset < count:
        end = min(offset + config.batch_size, count)
        batch = np.ascontiguousarray(vectors[offset:end], dtype=np.float32)
        ids = np.arange(offset, end, dtype=np.int64)
        index.add_with_ids(batch, ids)
        offset = end
    add_time_s = time.perf_counter() - add_started

    index_path = config.output_dir / f"new_custom_{config.index_type}.faiss"
    faiss.write_index(index, str(index_path))
    index_size_mb = index_path.stat().st_size / 1024 / 1024

    query_count = min(100, count)
    latency_ms = None
    if query_count > 0:
        queries = np.ascontiguousarray(vectors[:query_count], dtype=np.float32)
        latency_started = time.perf_counter()
        index.search(queries, 10)
        latency_ms = ((time.perf_counter() - latency_started) * 1000) / query_count

    return {
        "index_path": str(index_path),
        "index_type": config.index_type,
        "params": params,
        "vectors": count,
        "train_time_s": round(train_time_s, 3),
        "add_time_s": round(add_time_s, 3),
        "total_build_time_s": round(time.perf_counter() - started, 3),
        "index_size_mb": round(index_size_mb, 3),
        "search_latency_ms_per_query_sample": round(latency_ms, 3) if latency_ms is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract new custom embeddings from image folders and build a separate scale FAISS index."
    )
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--model-name", default=EXPECTED_MODEL_NAME)
    parser.add_argument("--output-dir", default="reports/new_custom_scale")
    parser.add_argument("--max-images", type=positive_int, default=1_000_000)
    parser.add_argument("--batch-size", type=batch_size_value, default=4096)
    parser.add_argument("--index-type", choices=["flat", "ivfpq"], default="ivfpq")
    parser.add_argument("--nlist", type=positive_int, default=4096)
    parser.add_argument("--m", type=positive_int, default=32)
    parser.add_argument("--nbits", type=positive_int, default=8)
    parser.add_argument("--nprobe", type=positive_int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ScaleConfig(
        sources=[resolve_path(item) for item in args.sources],
        model_name=args.model_name,
        output_dir=resolve_path(args.output_dir),
        max_images=args.max_images,
        batch_size=args.batch_size,
        index_type=args.index_type,
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
        nprobe=args.nprobe,
        dry_run=args.dry_run,
    )
    custom_settings = custom_runtime_settings(settings)
    validate_runtime(custom_settings, config.model_name)

    image_paths = discover_images(config.sources, config.max_images)
    if not image_paths:
        raise SystemExit("No image files found in --sources")
    if config.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "sources": [str(path) for path in config.sources],
                    "images_found": len(image_paths),
                    "output_dir": str(config.output_dir),
                    "model_name": config.model_name,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    started = time.perf_counter()
    memmap_path, mapping_path, extract_summary = extract_embeddings(
        image_paths,
        config=config,
        custom_settings=custom_settings,
    )
    index_summary = build_index(
        memmap_path,
        count=extract_summary.embeddings_created,
        config=config,
    )
    summary = {
        "model_name": config.model_name,
        "sources": [str(path) for path in config.sources],
        "total_images_seen": extract_summary.images_seen,
        "embeddings_created": extract_summary.embeddings_created,
        "skipped_no_face": extract_summary.skipped_no_face,
        "skipped_invalid": extract_summary.skipped_invalid,
        "skipped_multiple_faces": extract_summary.skipped_multiple_faces,
        "errors": extract_summary.errors,
        "extract_time_s": round(extract_summary.extract_time_s, 3),
        "embedding_memmap_path": str(memmap_path),
        "mapping_path": str(mapping_path),
        **index_summary,
        "total_time_s": round(time.perf_counter() - started, 3),
    }
    write_json(config.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
