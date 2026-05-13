"""Large-scale synthetic FAISS retrieval benchmark.

This script is intentionally separate from ``scripts/benchmark_retrieval.py``.
It is designed for 1M/2M synthetic-vector experiments where holding raw vectors,
Flat ground-truth indexes, benchmark indexes, and serialization buffers in RAM
at the same time is unsafe.

The benchmark uses memmapped L2-normalized vectors, computes exact top-K
ground truth blockwise, builds one FAISS method at a time, and writes JSON,
CSV, and Markdown outputs. The outputs are benchmark artifacts only after an
actual local run; synthetic retrieval results are not biometric accuracy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

DEFAULT_K_VALUES = [1, 5, 10]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def mb_from_bytes(value: int | float) -> float:
    return round(float(value) / 1024 / 1024, 6)


def environment_snapshot() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "faiss_threads": int(faiss.omp_get_max_threads()),
    }


def ensure_clean_output_dirs(output_dir: Path, tmp_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, np.maximum(norms, 1e-12), out=vectors)
    return vectors


def generate_memmap_vectors(
    *,
    path: Path,
    n_vectors: int,
    dim: int,
    seed: int,
    batch_size: int,
) -> np.memmap:
    """Generate L2-normalized synthetic vectors into a float32 memmap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    vectors = np.memmap(path, dtype=np.float32, mode="w+", shape=(n_vectors, dim))
    rng = np.random.RandomState(seed)
    for start in range(0, n_vectors, batch_size):
        end = min(start + batch_size, n_vectors)
        block = rng.standard_normal((end - start, dim)).astype(np.float32)
        normalize_rows(block)
        vectors[start:end] = block
        vectors.flush()
    return vectors


def reopen_memmap(path: Path, n_vectors: int, dim: int, mode: str = "r") -> np.memmap:
    return np.memmap(path, dtype=np.float32, mode=mode, shape=(n_vectors, dim))


def select_query_vectors(
    vectors: np.memmap,
    *,
    n_queries: int,
    query_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    actual_queries = min(n_queries, vectors.shape[0])
    indices = np.random.RandomState(query_seed).choice(
        vectors.shape[0],
        size=actual_queries,
        replace=False,
    )
    queries = np.asarray(vectors[indices], dtype=np.float32).copy()
    normalize_rows(queries)
    return indices.astype(np.int64), queries


def exact_topk_blockwise(
    vectors: np.memmap,
    queries: np.ndarray,
    *,
    max_k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact inner-product top-K without building a Flat FAISS index."""
    if max_k <= 0:
        raise ValueError("max_k must be positive")
    if max_k > vectors.shape[0]:
        raise ValueError("max_k must be <= number of vectors")

    query_count = queries.shape[0]
    top_scores = np.full((query_count, 0), -np.inf, dtype=np.float32)
    top_ids = np.full((query_count, 0), -1, dtype=np.int64)

    for start in range(0, vectors.shape[0], batch_size):
        end = min(start + batch_size, vectors.shape[0])
        block = np.asarray(vectors[start:end], dtype=np.float32)
        scores = queries @ block.T
        ids = np.arange(start, end, dtype=np.int64)
        ids = np.broadcast_to(ids, scores.shape)

        merged_scores = np.concatenate([top_scores, scores], axis=1)
        merged_ids = np.concatenate([top_ids, ids], axis=1)

        if merged_scores.shape[1] > max_k:
            selected = np.argpartition(-merged_scores, kth=max_k - 1, axis=1)[:, :max_k]
        else:
            selected = np.tile(
                np.arange(merged_scores.shape[1], dtype=np.int64),
                (query_count, 1),
            )

        top_scores = np.take_along_axis(merged_scores, selected, axis=1)
        top_ids = np.take_along_axis(merged_ids, selected, axis=1)

        order = np.argsort(-top_scores, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        top_ids = np.take_along_axis(top_ids, order, axis=1)

    return top_scores, top_ids


def compute_top_k_overlap(
    exact_ids: np.ndarray,
    candidate_ids: np.ndarray,
    k_values: list[int],
) -> dict[str, float]:
    results: dict[str, float] = {}
    query_count = exact_ids.shape[0]
    if query_count == 0:
        return {f"top_k_overlap@{k}": 0.0 for k in k_values}

    for k in k_values:
        hits = 0.0
        for query_idx in range(query_count):
            exact_set = {int(item) for item in exact_ids[query_idx, :k] if item >= 0}
            candidate_set = {int(item) for item in candidate_ids[query_idx, :k] if item >= 0}
            hits += len(exact_set & candidate_set) / k
        results[f"top_k_overlap@{k}"] = round(hits / query_count, 6)
    return results


def latency_percentiles(timings_ms: list[float]) -> dict[str, float]:
    if not timings_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    values = np.asarray(timings_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(values)), 6),
        "p50_ms": round(float(np.percentile(values, 50)), 6),
        "p95_ms": round(float(np.percentile(values, 95)), 6),
        "p99_ms": round(float(np.percentile(values, 99)), 6),
    }


def benchmark_flat_blockwise(
    vectors: np.memmap,
    queries: np.ndarray,
    exact_ids: np.ndarray,
    *,
    k_values: list[int],
    batch_size: int,
    latency_warmup: int,
    latency_runs: int,
) -> dict[str, Any]:
    max_k = max(k_values)
    for run_idx in range(latency_warmup):
        query = queries[run_idx % len(queries) : run_idx % len(queries) + 1]
        exact_topk_blockwise(vectors, query, max_k=max_k, batch_size=batch_size)

    timings_ms: list[float] = []
    candidate_ids = np.empty((len(queries), max_k), dtype=np.int64)
    for run_idx in range(latency_runs):
        query_idx = run_idx % len(queries)
        query = queries[query_idx : query_idx + 1]
        started = time.perf_counter_ns()
        _scores, ids = exact_topk_blockwise(vectors, query, max_k=max_k, batch_size=batch_size)
        timings_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        if run_idx < len(queries):
            candidate_ids[query_idx] = ids[0]

    if latency_runs < len(queries):
        _scores, candidate_ids = exact_topk_blockwise(
            vectors,
            queries,
            max_k=max_k,
            batch_size=batch_size,
        )

    return {
        "method": "flat_blockwise",
        "params": {"metric": "inner_product", "normalized_vectors": True},
        "status": "ok",
        "build_time_s": 0.0,
        "index_file_size_mb": None,
        "memory_estimate_mb": mb_from_bytes(vectors.size * np.dtype(np.float32).itemsize),
        "latency": latency_percentiles(timings_ms),
        "top_k_overlap": compute_top_k_overlap(exact_ids, candidate_ids, k_values),
        "notes": "Exact blockwise baseline; no FAISS Flat index is materialized.",
        "error": "",
    }


def add_vectors_in_batches(index: faiss.Index, vectors: np.memmap, batch_size: int) -> None:
    for start in range(0, vectors.shape[0], batch_size):
        end = min(start + batch_size, vectors.shape[0])
        block = np.ascontiguousarray(vectors[start:end], dtype=np.float32)
        index.add(block)


def sample_training_vectors(
    vectors: np.memmap,
    *,
    train_size: int,
    seed: int,
) -> np.ndarray:
    actual_size = min(train_size, vectors.shape[0])
    indices = np.random.RandomState(seed).choice(
        vectors.shape[0],
        size=actual_size,
        replace=False,
    )
    train_vectors = np.asarray(vectors[indices], dtype=np.float32).copy()
    normalize_rows(train_vectors)
    return train_vectors


def build_hnsw_index(vectors: np.memmap, args: argparse.Namespace) -> tuple[faiss.Index, dict[str, Any]]:
    index = faiss.IndexHNSWFlat(vectors.shape[1], args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = args.hnsw_ef_construction
    add_vectors_in_batches(index, vectors, args.batch_size)
    index.hnsw.efSearch = args.hnsw_ef_search
    params = {
        "M": args.hnsw_m,
        "efConstruction": args.hnsw_ef_construction,
        "efSearch": args.hnsw_ef_search,
        "metric": "inner_product",
        "normalized_vectors": True,
    }
    return index, params


def build_ivfpq_index(
    vectors: np.memmap,
    args: argparse.Namespace,
) -> tuple[faiss.Index, dict[str, Any], str]:
    notes: list[str] = []
    if vectors.shape[1] % args.ivfpq_m != 0:
        raise ValueError(
            f"embedding dimension {vectors.shape[1]} must be divisible by IVFPQ_M={args.ivfpq_m}"
        )

    train_vectors = sample_training_vectors(vectors, train_size=args.train_size, seed=args.seed + 1009)
    effective_nlist = min(args.ivfpq_nlist, len(train_vectors))
    if effective_nlist < args.ivfpq_nlist:
        notes.append(f"reduced nlist from {args.ivfpq_nlist} to {effective_nlist}")

    recommended = 39 * max(effective_nlist, 2**args.ivfpq_nbits)
    if len(train_vectors) < recommended:
        notes.append(
            "FAISS clustering warning: training sample is smaller than the usual "
            f"recommendation ({len(train_vectors)} < {recommended})"
        )

    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFPQ(
        quantizer,
        vectors.shape[1],
        effective_nlist,
        args.ivfpq_m,
        args.ivfpq_nbits,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.train(train_vectors)
    add_vectors_in_batches(index, vectors, args.batch_size)
    index.nprobe = min(args.ivfpq_nprobe, effective_nlist)

    params = {
        "nlist": effective_nlist,
        "requested_nlist": args.ivfpq_nlist,
        "M_pq": args.ivfpq_m,
        "nbits": args.ivfpq_nbits,
        "nprobe": int(index.nprobe),
        "train_size": len(train_vectors),
        "metric": "inner_product",
        "normalized_vectors": True,
    }
    return index, params, "; ".join(notes)


def estimate_index_file_size_mb(
    index: faiss.Index,
    *,
    tmp_dir: Path,
    method: str,
    n_vectors: int,
    keep_index_files: bool,
) -> float:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    index_path = tmp_dir / f"{method}_{n_vectors}_{os.getpid()}.faiss"
    faiss.write_index(index, str(index_path))
    size_mb = mb_from_bytes(index_path.stat().st_size)
    if not keep_index_files:
        index_path.unlink(missing_ok=True)
    return size_mb


def search_index(index: faiss.Index, queries: np.ndarray, max_k: int) -> np.ndarray:
    _scores, ids = index.search(np.ascontiguousarray(queries, dtype=np.float32), max_k)
    return ids.astype(np.int64, copy=False)


def measure_faiss_latency(
    index: faiss.Index,
    queries: np.ndarray,
    *,
    max_k: int,
    latency_warmup: int,
    latency_runs: int,
) -> dict[str, float]:
    for run_idx in range(latency_warmup):
        query = queries[run_idx % len(queries) : run_idx % len(queries) + 1]
        index.search(query, max_k)

    timings_ms: list[float] = []
    for run_idx in range(latency_runs):
        query = queries[run_idx % len(queries) : run_idx % len(queries) + 1]
        started = time.perf_counter_ns()
        index.search(query, max_k)
        timings_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    return latency_percentiles(timings_ms)


def benchmark_faiss_method(
    *,
    method: str,
    vectors: np.memmap,
    queries: np.ndarray,
    exact_ids: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    max_k = max(args.k_values)
    started = time.perf_counter()
    notes = ""
    index: faiss.Index | None = None
    try:
        if method == "hnsw":
            index, params = build_hnsw_index(vectors, args)
        elif method == "ivfpq":
            index, params, notes = build_ivfpq_index(vectors, args)
        else:
            raise ValueError(f"unsupported FAISS method: {method}")
        build_time_s = time.perf_counter() - started

        candidate_ids = search_index(index, queries, max_k)
        latency = measure_faiss_latency(
            index,
            queries,
            max_k=max_k,
            latency_warmup=args.latency_warmup,
            latency_runs=args.latency_runs,
        )
        index_file_size_mb = estimate_index_file_size_mb(
            index,
            tmp_dir=args.tmp_dir,
            method=method,
            n_vectors=vectors.shape[0],
            keep_index_files=args.keep_index_files,
        )
        return {
            "method": method,
            "params": params,
            "status": "ok",
            "build_time_s": round(build_time_s, 6),
            "index_file_size_mb": index_file_size_mb,
            "memory_estimate_mb": index_file_size_mb,
            "latency": latency,
            "top_k_overlap": compute_top_k_overlap(exact_ids, candidate_ids, args.k_values),
            "notes": notes,
            "error": "",
        }
    finally:
        del index
        gc.collect()


def failed_row(method: str, error: Exception) -> dict[str, Any]:
    return {
        "method": method,
        "params": {},
        "status": "failed",
        "build_time_s": None,
        "index_file_size_mb": None,
        "memory_estimate_mb": None,
        "latency": {"mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None},
        "top_k_overlap": {f"top_k_overlap@{k}": None for k in DEFAULT_K_VALUES},
        "notes": "",
        "error": str(error),
    }


def benchmark_size(n_vectors: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    vector_path = args.tmp_dir / f"synthetic_n{n_vectors}_d{args.dim}_seed{args.seed}.dat"
    vectors = generate_memmap_vectors(
        path=vector_path,
        n_vectors=n_vectors,
        dim=args.dim,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    vectors = reopen_memmap(vector_path, n_vectors=n_vectors, dim=args.dim)
    _query_indices, queries = select_query_vectors(
        vectors,
        n_queries=args.n_queries,
        query_seed=args.query_seed,
    )

    max_k = max(args.k_values)
    print(f"Computing blockwise exact top-{max_k} for N={n_vectors} ...")
    _exact_scores, exact_ids = exact_topk_blockwise(
        vectors,
        queries,
        max_k=max_k,
        batch_size=args.batch_size,
    )

    rows: list[dict[str, Any]] = []
    common = {
        "source": "synthetic_memmap",
        "n_vectors": n_vectors,
        "dim": args.dim,
        "n_queries": len(queries),
        "seed": args.seed,
        "query_seed": args.query_seed,
        "k_values": args.k_values,
        "batch_size": args.batch_size,
    }

    for method in args.methods:
        print(f"Benchmarking {method} for N={n_vectors} ...")
        try:
            if method == "flat":
                row = benchmark_flat_blockwise(
                    vectors,
                    queries,
                    exact_ids,
                    k_values=args.k_values,
                    batch_size=args.batch_size,
                    latency_warmup=args.latency_warmup,
                    latency_runs=args.latency_runs,
                )
            else:
                row = benchmark_faiss_method(
                    method=method,
                    vectors=vectors,
                    queries=queries,
                    exact_ids=exact_ids,
                    args=args,
                )
        except Exception as exc:  # noqa: BLE001
            row = failed_row(method, exc)
        row.update(common)
        rows.append(row)
        gc.collect()

    del vectors
    gc.collect()
    if not args.keep_vectors:
        vector_path.unlink(missing_ok=True)
    return rows


def params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"meta": meta, "results": rows}, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "source",
        "n_vectors",
        "dim",
        "n_queries",
        "seed",
        "query_seed",
        "method",
        "status",
        "params",
        "build_time_s",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "memory_estimate_mb",
        "index_file_size_mb",
        "top_k_overlap_at_1",
        "top_k_overlap_at_5",
        "top_k_overlap_at_10",
        "notes",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            latency = row["latency"]
            overlap = row["top_k_overlap"]
            writer.writerow(
                {
                    "source": row["source"],
                    "n_vectors": row["n_vectors"],
                    "dim": row["dim"],
                    "n_queries": row["n_queries"],
                    "seed": row["seed"],
                    "query_seed": row["query_seed"],
                    "method": row["method"],
                    "status": row["status"],
                    "params": params_json(row["params"]),
                    "build_time_s": row["build_time_s"],
                    "mean_ms": latency["mean_ms"],
                    "p50_ms": latency["p50_ms"],
                    "p95_ms": latency["p95_ms"],
                    "p99_ms": latency["p99_ms"],
                    "memory_estimate_mb": row["memory_estimate_mb"],
                    "index_file_size_mb": row["index_file_size_mb"],
                    "top_k_overlap_at_1": overlap.get("top_k_overlap@1"),
                    "top_k_overlap_at_5": overlap.get("top_k_overlap@5"),
                    "top_k_overlap_at_10": overlap.get("top_k_overlap@10"),
                    "notes": row.get("notes", ""),
                    "error": row.get("error", ""),
                }
            )


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    env = meta["environment"]
    config = meta["config"]
    lines = [
        "# Large-Scale Synthetic Retrieval Benchmark Results",
        "",
        f"Generated at: `{env['timestamp_utc']}`",
        "",
        "## Methodology",
        "",
        "- Synthetic vectors are generated as L2-normalized float32 embeddings.",
        "- Base vectors are stored in an ignored `np.memmap` file under the configured temporary directory.",
        "- Exact top-K ground truth is computed blockwise with inner-product similarity.",
        "- FAISS methods are built and benchmarked one at a time to reduce peak memory usage.",
        "- Index size is measured by writing a temporary FAISS index file and measuring file size.",
        "- Temporary index files are deleted unless `--keep-index-files` is passed.",
        "",
        "## Metric Definition",
        "",
        "`top_k_overlap@K = |exact_top_K(query) intersect approximate_top_K(query)| / K`",
        "",
        "This is not biometric identification hit@K and not biometric accuracy. "
        "The benchmark uses synthetic vectors with no identity labels, so it evaluates vector retrieval behavior only.",
        "",
        "## Hardware/Environment",
        "",
        f"- Platform: `{env['platform']}`",
        f"- Processor: `{env['processor']}`",
        f"- Machine: `{env['machine']}`",
        f"- CPU count: `{env['cpu_count']}`",
        f"- Python: `{env['python']}`",
        f"- NumPy: `{env['numpy']}`",
        f"- FAISS: `{env['faiss']}`",
        f"- FAISS threads: `{env['faiss_threads']}`",
        "- Full process RSS is not measured; memory estimate is based on temporary FAISS index file size.",
        "",
        "## Run Configuration",
        "",
        f"- Sizes: `{', '.join(str(item) for item in config['sizes'])}`",
        f"- Embedding dimension: `{config['dim']}`",
        f"- Queries: `{config['n_queries']}`",
        f"- Seed: `{config['seed']}`",
        f"- Query seed: `{config['query_seed']}`",
        f"- Batch size: `{config['batch_size']}`",
        f"- Train size: `{config['train_size']}`",
        f"- Methods: `{', '.join(config['methods'])}`",
        f"- K values: `{', '.join(str(item) for item in config['k_values'])}`",
        "",
        "## Results",
        "",
        "| N | Dim | Queries | Method | Status | Build time (s) | Mean ms | p50 ms | p95 ms | p99 ms | Memory MB | Index file MB | top_k_overlap@1 | top_k_overlap@5 | top_k_overlap@10 | Notes |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        latency = row["latency"]
        overlap = row["top_k_overlap"]
        notes = row.get("notes") or row.get("error") or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["n_vectors"]),
                    str(row["dim"]),
                    str(row["n_queries"]),
                    row["method"],
                    row["status"],
                    fmt(row["build_time_s"]),
                    fmt(latency["mean_ms"]),
                    fmt(latency["p50_ms"]),
                    fmt(latency["p95_ms"]),
                    fmt(latency["p99_ms"]),
                    fmt(row["memory_estimate_mb"]),
                    fmt(row["index_file_size_mb"]),
                    fmt(overlap.get("top_k_overlap@1")),
                    fmt(overlap.get("top_k_overlap@5")),
                    fmt(overlap.get("top_k_overlap@10")),
                    notes.replace("|", "\\|"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The benchmark uses synthetic L2-normalized vectors, not real biometric identities.",
            "- It does not measure FAR, FRR, EER, liveness resistance, API latency, detector latency, or UI latency.",
            "- `top_k_overlap@K` measures agreement with exact blockwise top-K retrieval, not recognition accuracy.",
            "- Results should only be reported after the script is actually run on the target hardware.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nSUMMARY - SCALE RETRIEVAL BENCHMARK")
    print(
        f"{'N':>10} {'Method':<16} {'Status':<8} {'O@1':>8} {'O@5':>8} {'O@10':>8} "
        f"{'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10} {'Build s':>10} {'MB':>10}"
    )
    print("-" * 126)
    for row in rows:
        overlap = row["top_k_overlap"]
        latency = row["latency"]
        print(
            f"{row['n_vectors']:>10} {row['method']:<16} {row['status']:<8} "
            f"{fmt(overlap.get('top_k_overlap@1'), 4):>8} "
            f"{fmt(overlap.get('top_k_overlap@5'), 4):>8} "
            f"{fmt(overlap.get('top_k_overlap@10'), 4):>8} "
            f"{fmt(latency['p50_ms'], 4):>10} "
            f"{fmt(latency['p95_ms'], 4):>10} "
            f"{fmt(latency['p99_ms'], 4):>10} "
            f"{fmt(row['build_time_s'], 4):>10} "
            f"{fmt(row['memory_estimate_mb'], 4):>10}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Large-scale synthetic FAISS retrieval benchmark"
    )
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=[100000])
    parser.add_argument("--dim", type=positive_int, default=512)
    parser.add_argument("--n-queries", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-seed", type=int, default=123)
    parser.add_argument(
        "--methods",
        choices=["flat", "hnsw", "ivfpq"],
        nargs="+",
        default=["hnsw", "ivfpq"],
    )
    parser.add_argument("--k-values", type=positive_int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--output-dir", type=Path, default=Path("training/outputs/scale_benchmark"))
    parser.add_argument("--tmp-dir", type=Path, default=Path("tmp/scale_benchmark"))
    parser.add_argument("--batch-size", type=positive_int, default=50000)
    parser.add_argument("--train-size", type=positive_int, default=100000)
    parser.add_argument("--latency-warmup", type=positive_int, default=5)
    parser.add_argument("--latency-runs", type=positive_int, default=100)
    parser.add_argument("--faiss-threads", type=positive_int, default=None)
    parser.add_argument("--keep-index-files", action="store_true")
    parser.add_argument("--keep-vectors", action="store_true")

    parser.add_argument("--hnsw-m", type=positive_int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=positive_int, default=100)
    parser.add_argument("--hnsw-ef-search", type=positive_int, default=64)

    parser.add_argument("--ivfpq-nlist", type=positive_int, default=4096)
    parser.add_argument("--ivfpq-m", type=positive_int, default=16)
    parser.add_argument("--ivfpq-nbits", type=positive_int, default=8)
    parser.add_argument("--ivfpq-nprobe", type=positive_int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.k_values = sorted(set(args.k_values))
    if args.faiss_threads is not None:
        faiss.omp_set_num_threads(args.faiss_threads)
    ensure_clean_output_dirs(args.output_dir, args.tmp_dir)

    rows: list[dict[str, Any]] = []
    for n_vectors in args.sizes:
        print(f"Generating and benchmarking N={n_vectors}, dim={args.dim} ...")
        rows.extend(benchmark_size(n_vectors, args))

    meta = {
        "environment": environment_snapshot(),
        "config": {
            "sizes": args.sizes,
            "dim": args.dim,
            "n_queries": args.n_queries,
            "seed": args.seed,
            "query_seed": args.query_seed,
            "methods": args.methods,
            "k_values": args.k_values,
            "output_dir": str(args.output_dir),
            "tmp_dir": str(args.tmp_dir),
            "batch_size": args.batch_size,
            "train_size": args.train_size,
            "latency_warmup": args.latency_warmup,
            "latency_runs": args.latency_runs,
            "hnsw": {
                "M": args.hnsw_m,
                "efConstruction": args.hnsw_ef_construction,
                "efSearch": args.hnsw_ef_search,
            },
            "ivfpq": {
                "nlist": args.ivfpq_nlist,
                "M_pq": args.ivfpq_m,
                "nbits": args.ivfpq_nbits,
                "nprobe": args.ivfpq_nprobe,
            },
        },
    }

    json_path = args.output_dir / "retrieval_scale_results.json"
    csv_path = args.output_dir / "retrieval_scale_results.csv"
    markdown_path = args.output_dir / "retrieval_scale_results.md"
    write_json(json_path, meta, rows)
    write_csv(csv_path, rows)
    write_markdown(markdown_path, meta, rows)
    print_summary(rows)
    print(f"\nJSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
