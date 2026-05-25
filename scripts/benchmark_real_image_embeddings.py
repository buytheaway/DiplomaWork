"""Real-image embedding retrieval benchmark for the final custom model.

This benchmark extracts embeddings from real image files and builds a separate
FAISS index under the report directory. It does not use synthetic vectors and
does not touch the production database or runtime FAISS snapshots.
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
class BenchmarkConfig:
    sources: list[Path]
    output_dir: Path
    max_images: int
    batch_size: int
    n_queries: int
    top_k: int
    index_type: str
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    ivfpq_nlist: int
    ivfpq_m: int
    ivfpq_nbits: int
    ivfpq_nprobe: int
    lfw_metrics: Path


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


def validate_final_custom_runtime(custom_settings: Settings) -> None:
    if custom_settings.embedding_backend != "torch":
        raise RuntimeError("CUSTOM_BACKEND must resolve to torch")
    if custom_settings.torch_model_arch != "insightface_iresnet100":
        raise RuntimeError("TORCH_MODEL_ARCH must be insightface_iresnet100")
    if custom_settings.torch_preprocess != "runtime_fallback_center_crop":
        raise RuntimeError("TORCH_PREPROCESS must be runtime_fallback_center_crop")
    if custom_settings.torch_tta != "hflip":
        raise RuntimeError("TORCH_TTA must be hflip")


def discover_images(sources: list[Path], *, limit: int | None = None) -> tuple[list[Path], int]:
    selected: list[Path] = []
    total = 0

    def add_image(path: Path) -> None:
        nonlocal total
        total += 1
        if limit is None or len(selected) < limit:
            selected.append(path.resolve())

    for source in sources:
        if not source.exists():
            continue
        if source.is_file() and source.suffix.lower() in SUPPORTED_EXTENSIONS:
            add_image(source)
            continue
        if source.is_dir():
            for path in source.rglob("*"):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    add_image(path)
    return selected, total


def infer_identity(path: Path, sources: list[Path]) -> str:
    for source in sources:
        if source.is_dir():
            try:
                relative = path.relative_to(source)
            except ValueError:
                continue
            if (
                len(relative.parts) > 2
                and relative.parts[0].lower() in {"train", "val", "test"}
            ):
                return relative.parts[1]
            return relative.parts[0] if len(relative.parts) > 1 else source.name
    return path.parent.name


def read_lfw_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected_threshold = 0.205047
    far_at_selected = None
    frr_at_selected = None
    if abs(float(payload.get("best_accuracy_threshold", -1.0)) - selected_threshold) < 1e-9:
        far_at_selected = payload.get("far_at_best_threshold")
        frr_at_selected = payload.get("frr_at_best_threshold")
    return {
        "available": True,
        "path": str(path),
        "valid_pairs": payload.get("valid_pairs"),
        "skipped_pairs": payload.get("skipped_pairs"),
        "accuracy": payload.get("best_accuracy"),
        "eer": payload.get("eer"),
        "eer_threshold": payload.get("eer_threshold"),
        "far_at_best_threshold": payload.get("far_at_best_threshold"),
        "frr_at_best_threshold": payload.get("frr_at_best_threshold"),
        "selected_threshold": selected_threshold,
        "far_at_selected_threshold": far_at_selected,
        "frr_at_selected_threshold": frr_at_selected,
        "tar_at_far_0_1": payload.get("tar_at_far", {}).get("0.1", {}).get("tar"),
        "tar_at_far_0_01": payload.get("tar_at_far", {}).get("0.01", {}).get("tar"),
        "tar_at_far_0_001": payload.get("tar_at_far", {}).get("0.001", {}).get("tar"),
    }


def extract_real_embeddings(
    selected_images: list[Path],
    *,
    all_sources: list[Path],
    output_dir: Path,
    batch_size: int,
    custom_settings: Settings,
) -> tuple[Path, Path, dict[str, Any]]:
    extractor = create_extractor(custom_settings)
    if extractor.model_name != EXPECTED_MODEL_NAME:
        raise RuntimeError(f"Expected model {EXPECTED_MODEL_NAME}, got {extractor.model_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = output_dir / "real_image_embeddings.float32.memmap"
    mapping_path = output_dir / "real_image_mapping.csv"
    vectors = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(selected_images), 512),
    )

    stats = {
        "images_attempted": len(selected_images),
        "embeddings_created": 0,
        "skipped_no_face": 0,
        "skipped_invalid": 0,
        "skipped_multiple_faces": 0,
        "errors": 0,
    }
    started = time.perf_counter()
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "identity", "image_path"],
        )
        writer.writeheader()
        row_id = 0
        for image_path in selected_images:
            try:
                vector = extractor.extract_embedding(image_path.read_bytes())
            except NoFaceDetectedError:
                stats["skipped_no_face"] += 1
                continue
            except MultipleFacesDetectedError:
                stats["skipped_multiple_faces"] += 1
                continue
            except InvalidImageError:
                stats["skipped_invalid"] += 1
                continue
            except Exception:  # noqa: BLE001
                stats["errors"] += 1
                continue

            if vector.shape != (512,) or not np.isfinite(vector).all():
                stats["errors"] += 1
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0:
                stats["errors"] += 1
                continue
            vectors[row_id] = (vector / norm).astype(np.float32)
            writer.writerow(
                {
                    "row_id": row_id,
                    "identity": infer_identity(image_path, all_sources),
                    "image_path": str(image_path),
                }
            )
            row_id += 1
            stats["embeddings_created"] += 1
            if row_id % batch_size == 0:
                vectors.flush()

    vectors.flush()
    del vectors
    stats["extract_time_s"] = round(time.perf_counter() - started, 3)
    return memmap_path, mapping_path, stats


def build_faiss_index(
    memmap_path: Path,
    *,
    count: int,
    output_dir: Path,
    config: BenchmarkConfig,
) -> tuple[Path, dict[str, Any]]:
    if count <= 0:
        raise RuntimeError("No real-image embeddings were created")
    vectors = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(count, 512))
    started = time.perf_counter()
    train_time_s = 0.0

    if config.index_type == "hnsw":
        base = faiss.IndexHNSWFlat(512, config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = config.hnsw_ef_construction
        base.hnsw.efSearch = config.hnsw_ef_search
        index: faiss.Index = faiss.IndexIDMap2(base)
        params = {
            "m": config.hnsw_m,
            "ef_construction": config.hnsw_ef_construction,
            "ef_search": config.hnsw_ef_search,
        }
    elif config.index_type == "ivfpq":
        if 512 % config.ivfpq_m != 0:
            raise RuntimeError("Embedding dim 512 must be divisible by IVFPQ_M")
        train_size = min(count, max(config.ivfpq_nlist, min(count, 100_000)))
        train_vectors = np.ascontiguousarray(vectors[:train_size], dtype=np.float32)
        effective_nlist = min(config.ivfpq_nlist, train_size)
        quantizer = faiss.IndexFlatIP(512)
        base = faiss.IndexIVFPQ(
            quantizer,
            512,
            effective_nlist,
            config.ivfpq_m,
            config.ivfpq_nbits,
            faiss.METRIC_INNER_PRODUCT,
        )
        train_started = time.perf_counter()
        base.train(train_vectors)
        train_time_s = time.perf_counter() - train_started
        base.nprobe = min(config.ivfpq_nprobe, effective_nlist)
        index = faiss.IndexIDMap2(base)
        params = {
            "nlist": effective_nlist,
            "requested_nlist": config.ivfpq_nlist,
            "m": config.ivfpq_m,
            "nbits": config.ivfpq_nbits,
            "nprobe": int(base.nprobe),
            "train_size": train_size,
        }
    elif config.index_type == "flat":
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(512))
        params = {}
    else:
        raise RuntimeError("--index-type must be hnsw, ivfpq or flat")

    add_started = time.perf_counter()
    offset = 0
    while offset < count:
        end = min(offset + config.batch_size, count)
        batch = np.ascontiguousarray(vectors[offset:end], dtype=np.float32)
        ids = np.arange(offset, end, dtype=np.int64)
        index.add_with_ids(batch, ids)
        offset = end
    add_time_s = time.perf_counter() - add_started

    index_path = output_dir / f"real_image_{config.index_type}.faiss"
    faiss.write_index(index, str(index_path))
    build_stats = {
        "index_type": config.index_type,
        "index_params": params,
        "build_time_s": round(time.perf_counter() - started, 3),
        "train_time_s": round(train_time_s, 3),
        "add_time_s": round(add_time_s, 3),
        "index_size_bytes": index_path.stat().st_size,
        "index_size_mb": round(index_path.stat().st_size / 1024 / 1024, 3),
    }
    return index_path, {"index": index, "vectors": vectors, **build_stats}


def benchmark_search(
    index: faiss.Index,
    vectors: np.memmap,
    *,
    count: int,
    n_queries: int,
    top_k: int,
    mapping_rows: dict[int, dict[str, str]],
) -> dict[str, Any]:
    actual_queries = min(n_queries, count)
    latencies: list[float] = []
    sample: dict[str, Any] = {}
    for row_id in range(actual_queries):
        query = np.ascontiguousarray(vectors[row_id : row_id + 1], dtype=np.float32)
        started = time.perf_counter()
        scores, ids = index.search(query, top_k)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        if row_id == 0:
            top_ids = [int(item) for item in ids[0].tolist()]
            sample = {
                "query_row_id": 0,
                "query_identity": mapping_rows.get(0, {}).get("identity"),
                "query_image_path": mapping_rows.get(0, {}).get("image_path"),
                "top_ids": top_ids,
                "top_scores": [float(item) for item in scores[0].tolist()],
                "top_results": [
                    {
                        "row_id": item,
                        "identity": mapping_rows.get(item, {}).get("identity"),
                        "image_path": mapping_rows.get(item, {}).get("image_path"),
                        "score": float(scores[0][position]),
                    }
                    for position, item in enumerate(top_ids)
                    if item >= 0
                ],
            }
    values = np.asarray(latencies, dtype=np.float64)
    return {
        "n_queries": actual_queries,
        "top_k": top_k,
        "p50_ms": round(float(np.percentile(values, 50)), 6),
        "p95_ms": round(float(np.percentile(values, 95)), 6),
        "p99_ms": round(float(np.percentile(values, 99)), 6),
        "mean_ms": round(float(values.mean()), 6),
        "sample_query_result": sample,
    }


def load_mapping_rows(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row_id = int(row["row_id"])
            except (KeyError, TypeError, ValueError):
                continue
            rows[row_id] = {
                "identity": row.get("identity", ""),
                "image_path": row.get("image_path", ""),
            }
    return rows


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lfw = summary["lfw_metrics"]
    lines = [
        "# Real-Image Embedding Retrieval Benchmark",
        "",
        "## Methodology",
        "",
        "This benchmark extracts embeddings from real face image files with the final custom Torch runtime model and builds a separate FAISS retrieval index. It does not use synthetic random vectors and does not touch the production database/index.",
        "",
        "Synthetic 1M/2M FAISS benchmarks remain scalability-only experiments and are not biometric accuracy measurements.",
        "",
        "## Runtime",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Model | `{summary['model_name']}` |",
        "| Preprocess | `runtime_fallback_center_crop` |",
        "| Color / normalization | `RGB`, `[-1, 1]` |",
        "| TTA | `hflip` |",
        "| Threshold | `0.205047` |",
        "",
        "## Real-Image Retrieval Results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total real images scanned | {summary['total_real_images_scanned']} |",
        f"| Images attempted | {summary['images_attempted']} |",
        f"| Embeddings created | {summary['embeddings_created']} |",
        f"| Skipped no-face | {summary['skipped_no_face']} |",
        f"| Skipped invalid | {summary['skipped_invalid']} |",
        f"| Skipped multiple-faces | {summary['skipped_multiple_faces']} |",
        f"| Extractor errors | {summary['errors']} |",
        f"| Embedding dim | {summary['embedding_dim']} |",
        f"| Index type | `{summary['index_type']}` |",
        f"| Index size MB | {summary['index_size_mb']} |",
        f"| Extraction time s | {summary['extract_time_s']} |",
        f"| Build time s | {summary['build_time_s']} |",
        f"| Search p50 ms | {summary['search_latency']['p50_ms']} |",
        f"| Search p95 ms | {summary['search_latency']['p95_ms']} |",
        f"| Search p99 ms | {summary['search_latency']['p99_ms']} |",
        "",
        "Output paths:",
        "",
        f"- embeddings: `{summary['embedding_memmap_path']}`",
        f"- mapping: `{summary['mapping_path']}`",
        f"- index: `{summary['index_path']}`",
        "",
        "Sample query result:",
        "",
        "```json",
        json.dumps(summary["search_latency"]["sample_query_result"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## LFW Biometric Verification",
        "",
        "LFW is a real biometric verification protocol. These metrics evaluate identity verification quality, not FAISS scalability.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Valid pairs | {lfw.get('valid_pairs')} |",
        f"| Skipped pairs | {lfw.get('skipped_pairs')} |",
        f"| Accuracy | {lfw.get('accuracy')} |",
        f"| EER | {lfw.get('eer')} |",
        f"| EER threshold | {lfw.get('eer_threshold')} |",
        f"| FAR at selected threshold | {lfw.get('far_at_selected_threshold')} |",
        f"| FRR at selected threshold | {lfw.get('frr_at_selected_threshold')} |",
        f"| TAR@FAR=0.1 | {lfw.get('tar_at_far_0_1')} |",
        f"| TAR@FAR=0.01 | {lfw.get('tar_at_far_0_01')} |",
        f"| TAR@FAR=0.001 | {lfw.get('tar_at_far_0_001')} |",
        "",
        "## Interpretation",
        "",
        "- LFW verification closes the biometric-metrics gap with real image pairs.",
        "- This real-image retrieval benchmark proves that retrieval measurements were also run on embeddings extracted from real images.",
        "- The synthetic 1M/2M benchmark should be described only as FAISS scalability testing, not biometric recognition accuracy.",
        "- Do not claim 1M real-image vectors unless a real dataset import actually creates 1M real-image embeddings.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real-image custom embeddings with FAISS")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["datasets/celeba_faces", "handoff_lfw_eval/lfw", "data/new_custom_enroll"],
    )
    parser.add_argument("--output-dir", default="reports/real_image_embedding_benchmark")
    parser.add_argument("--max-images", type=positive_int, default=10_000)
    parser.add_argument("--batch-size", type=positive_int, default=512)
    parser.add_argument("--n-queries", type=positive_int, default=100)
    parser.add_argument("--top-k", type=positive_int, default=10)
    parser.add_argument("--index-type", choices=["hnsw", "ivfpq", "flat"], default="hnsw")
    parser.add_argument("--hnsw-m", type=positive_int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=positive_int, default=100)
    parser.add_argument("--hnsw-ef-search", type=positive_int, default=64)
    parser.add_argument("--ivfpq-nlist", type=positive_int, default=4096)
    parser.add_argument("--ivfpq-m", type=positive_int, default=32)
    parser.add_argument("--ivfpq-nbits", type=positive_int, default=8)
    parser.add_argument("--ivfpq-nprobe", type=positive_int, default=32)
    parser.add_argument(
        "--lfw-metrics",
        default="reports/biometric_eval/candidate_runtime_fallback_center_crop_hflip/metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        sources=[resolve_path(item) for item in args.sources],
        output_dir=resolve_path(args.output_dir),
        max_images=args.max_images,
        batch_size=args.batch_size,
        n_queries=args.n_queries,
        top_k=args.top_k,
        index_type=args.index_type,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        ivfpq_nlist=args.ivfpq_nlist,
        ivfpq_m=args.ivfpq_m,
        ivfpq_nbits=args.ivfpq_nbits,
        ivfpq_nprobe=args.ivfpq_nprobe,
        lfw_metrics=resolve_path(args.lfw_metrics),
    )
    custom_settings = custom_runtime_settings(settings)
    validate_final_custom_runtime(custom_settings)

    selected_images, total_images = discover_images(config.sources, limit=config.max_images)
    if not selected_images:
        raise SystemExit("No real images found in configured sources")

    started = time.perf_counter()
    memmap_path, mapping_path, extract_stats = extract_real_embeddings(
        selected_images,
        all_sources=config.sources,
        output_dir=config.output_dir,
        batch_size=config.batch_size,
        custom_settings=custom_settings,
    )
    count = int(extract_stats["embeddings_created"])
    index_path, index_payload = build_faiss_index(
        memmap_path,
        count=count,
        output_dir=config.output_dir,
        config=config,
    )
    search_latency = benchmark_search(
        index_payload["index"],
        index_payload["vectors"],
        count=count,
        n_queries=config.n_queries,
        top_k=config.top_k,
        mapping_rows=load_mapping_rows(mapping_path),
    )

    summary = {
        "model_name": EXPECTED_MODEL_NAME,
        "sources": [str(path) for path in config.sources],
        "total_real_images_scanned": total_images,
        "max_images": config.max_images,
        "embedding_dim": 512,
        "embedding_memmap_path": str(memmap_path),
        "mapping_path": str(mapping_path),
        "index_path": str(index_path),
        **extract_stats,
        **{
            key: value
            for key, value in index_payload.items()
            if key not in {"index", "vectors"}
        },
        "search_latency": search_latency,
        "lfw_metrics": read_lfw_metrics(config.lfw_metrics),
        "total_runtime_s": round(time.perf_counter() - started, 3),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(summary, config.output_dir / "summary.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
