"""Benchmark FAISS retrieval methods: Flat vs HNSW vs IVF-PQ.

The script measures top-k overlap@1/5/10, single-query latency percentiles,
serialized index size, and index build time. It can run on generated
unit-normalized vectors or vectors reconstructed from an existing FAISS index.
Results are written as JSON, CSV, and Markdown so they can be used only after an
actual local run.

Usage:

    python scripts/benchmark_retrieval.py

    python scripts/benchmark_retrieval.py --source-index backend/data/index/custom.faiss

    python scripts/benchmark_retrieval.py --sizes 100 1000 10000 --dim 512 --n-queries 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

K_VALUES = [1, 5, 10]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_flat(vectors: np.ndarray) -> faiss.Index:
    """Build a flat exact inner-product index."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def build_hnsw(vectors: np.ndarray, hnsw_m: int, ef_construction: int) -> faiss.Index:
    """Build an HNSW inner-product index."""
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def build_ivfpq(vectors: np.ndarray, nlist: int, m_pq: int, nbits: int) -> faiss.Index:
    """Build an IVF-PQ inner-product index."""
    dim = vectors.shape[1]
    if dim % m_pq != 0:
        raise ValueError(f"embedding dimension {dim} must be divisible by IVFPQ_M={m_pq}")

    min_train_vectors = max(nlist, 2**nbits)
    if len(vectors) < min_train_vectors:
        raise ValueError(
            "not enough vectors for IVF-PQ training: "
            f"n_vectors={len(vectors)}, required_at_least={min_train_vectors}"
        )

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    return index


def measure_top_k_overlap(
    index: faiss.Index,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k_values: list[int],
) -> dict[str, float]:
    """Compute average top-k overlap with Flat exact search results."""
    max_k = max(k_values)
    _, ids = index.search(queries, max_k)

    results: dict[str, float] = {}
    for k in k_values:
        hits = 0.0
        for query_idx in range(len(queries)):
            gt_set = {int(item) for item in ground_truth[query_idx, :k] if item >= 0}
            pred_set = {int(item) for item in ids[query_idx, :k] if item >= 0}
            hits += len(gt_set & pred_set) / k
        results[f"top_k_overlap@{k}"] = round(hits / len(queries), 6)
    return results


def measure_latency(
    index: faiss.Index,
    queries: np.ndarray,
    k: int,
    n_warmup: int,
    n_runs: int,
) -> dict[str, float]:
    """Measure single-query FAISS search latency in milliseconds."""
    if len(queries) == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    for run_idx in range(n_warmup):
        query = queries[run_idx % len(queries) : run_idx % len(queries) + 1]
        index.search(query, k)

    timings_ms: list[float] = []
    for run_idx in range(n_runs):
        query = queries[run_idx % len(queries) : run_idx % len(queries) + 1]
        started = time.perf_counter_ns()
        index.search(query, k)
        timings_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    timings = np.array(timings_ms, dtype=np.float64)
    return {
        "mean": round(float(np.mean(timings)), 6),
        "p50": round(float(np.percentile(timings, 50)), 6),
        "p95": round(float(np.percentile(timings, 95)), 6),
        "p99": round(float(np.percentile(timings, 99)), 6),
    }


def measure_memory_mb(index: faiss.Index) -> float:
    """Estimate index memory by serializing the FAISS index to bytes."""
    writer = faiss.VectorIOWriter()
    faiss.write_index(index, writer)
    return round(len(faiss.vector_to_array(writer.data)) / 1024 / 1024, 6)


def generate_vectors(n_vectors: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def load_vectors_from_index(source_index_path: Path) -> np.ndarray:
    src_index = faiss.read_index(str(source_index_path))
    n_vectors = src_index.ntotal
    dim = src_index.d
    vectors = np.zeros((n_vectors, dim), dtype=np.float32)
    for index_id in range(n_vectors):
        vectors[index_id] = src_index.reconstruct(index_id)
    faiss.normalize_L2(vectors)
    return vectors


def select_queries(vectors: np.ndarray, n_queries: int, query_seed: int) -> np.ndarray:
    actual_queries = min(n_queries, len(vectors))
    query_indices = np.random.RandomState(query_seed).choice(
        len(vectors),
        size=actual_queries,
        replace=False,
    )
    return vectors[query_indices].copy()


def environment_snapshot() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "faiss_threads": int(faiss.omp_get_max_threads()),
    }


def benchmark_method(
    *,
    method: str,
    params: dict[str, Any],
    build_index: Any,
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    latency_warmup: int,
    latency_runs: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    index = build_index()
    build_time_s = time.perf_counter() - started

    top_k_overlap = measure_top_k_overlap(index, queries, ground_truth, K_VALUES)
    latency = measure_latency(
        index,
        queries,
        k=max(K_VALUES),
        n_warmup=latency_warmup,
        n_runs=latency_runs,
    )
    memory_mb = measure_memory_mb(index)

    return {
        "method": method,
        "params": params,
        "status": "ok",
        "build_time_s": round(build_time_s, 6),
        "memory_mb": memory_mb,
        "latency_ms": latency,
        "top_k_overlap": top_k_overlap,
    }


def benchmark_size(vectors: np.ndarray, args: argparse.Namespace, source_label: str) -> list[dict[str, Any]]:
    queries = select_queries(vectors, args.n_queries, args.query_seed)

    ground_truth_index = build_flat(vectors)
    _, ground_truth = ground_truth_index.search(queries, max(K_VALUES))

    rows: list[dict[str, Any]] = []
    common = {
        "source": source_label,
        "n_vectors": int(len(vectors)),
        "dim": int(vectors.shape[1]),
        "n_queries": int(len(queries)),
        "seed": int(args.seed),
        "query_seed": int(args.query_seed),
        "k_values": K_VALUES,
    }

    method_specs = [
        (
            "flat",
            {"metric": "inner_product", "normalized_vectors": True},
            lambda: build_flat(vectors),
            "",
        )
    ]

    for ef_search in args.hnsw_ef_search:
        params = {
            "M": args.hnsw_m,
            "efConstruction": args.hnsw_ef_construction,
            "efSearch": ef_search,
            "metric": "inner_product",
            "normalized_vectors": True,
        }

        def build_hnsw_with_params(ef_search: int = ef_search) -> faiss.Index:
            index = build_hnsw(vectors, args.hnsw_m, args.hnsw_ef_construction)
            index.hnsw.efSearch = ef_search
            return index

        method_specs.append((f"hnsw_ef{ef_search}", params, build_hnsw_with_params, ""))

    for nprobe in args.ivfpq_nprobe:
        params = {
            "nlist": args.ivfpq_nlist,
            "M_pq": args.ivfpq_m,
            "nbits": args.ivfpq_nbits,
            "nprobe": nprobe,
            "metric": "inner_product",
            "normalized_vectors": True,
        }

        def build_ivfpq_with_params(nprobe: int = nprobe) -> faiss.Index:
            index = build_ivfpq(vectors, args.ivfpq_nlist, args.ivfpq_m, args.ivfpq_nbits)
            index.nprobe = nprobe
            return index

        method_specs.append(
            (
                f"ivfpq_nprobe{nprobe}",
                params,
                build_ivfpq_with_params,
                ivfpq_training_note(len(vectors), params),
            )
        )

    for method, params, build_index, notes in method_specs:
        row = dict(common)
        try:
            result = benchmark_method(
                method=method,
                params=params,
                build_index=build_index,
                vectors=vectors,
                queries=queries,
                ground_truth=ground_truth,
                latency_warmup=args.latency_warmup,
                latency_runs=args.latency_runs,
            )
            row.update(result)
            row["notes"] = notes
        except Exception as exc:
            row.update(
                {
                    "method": method,
                    "params": params,
                    "status": "failed",
                    "error": str(exc),
                    "build_time_s": None,
                    "memory_mb": None,
                    "latency_ms": {"mean": None, "p50": None, "p95": None, "p99": None},
                    "top_k_overlap": {
                        "top_k_overlap@1": None,
                        "top_k_overlap@5": None,
                        "top_k_overlap@10": None,
                    },
                    "notes": notes,
                }
            )
        rows.append(row)
    return rows


def params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def ivfpq_training_note(n_vectors: int, params: dict[str, Any]) -> str:
    recommended = 39 * max(int(params["nlist"]), 2 ** int(params["nbits"]))
    if n_vectors >= recommended:
        return ""
    return (
        "FAISS clustering warning: this IVF-PQ configuration has fewer training "
        f"vectors than the usual recommendation ({n_vectors} < {recommended})."
    )


def write_json(path: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    payload = {"meta": meta, "results": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "memory_mb",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
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
            latency = row["latency_ms"]
            top_k_overlap = row["top_k_overlap"]
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
                    "memory_mb": row["memory_mb"],
                    "latency_mean_ms": latency["mean"],
                    "latency_p50_ms": latency["p50"],
                    "latency_p95_ms": latency["p95"],
                    "latency_p99_ms": latency["p99"],
                    "top_k_overlap_at_1": top_k_overlap["top_k_overlap@1"],
                    "top_k_overlap_at_5": top_k_overlap["top_k_overlap@5"],
                    "top_k_overlap_at_10": top_k_overlap["top_k_overlap@10"],
                    "notes": row.get("notes", ""),
                    "error": row.get("error", ""),
                }
            )


def format_value(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_params(params: dict[str, Any]) -> str:
    return "`" + params_json(params).replace("|", "\\|") + "`"


def write_markdown(path: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    env = meta["environment"]
    config = meta["config"]
    sizes = ", ".join(str(size) for size in config["sizes"])

    lines = [
        "# Retrieval Benchmark Results",
        "",
        f"Generated at: `{env['timestamp_utc']}`",
        "",
        "## Methodology",
        "",
        "- Generated vectors use NumPy RandomState unless a source FAISS index is provided.",
        "- All vectors are L2-normalized and searched with inner-product similarity.",
        "- Query vectors are sampled from the benchmark database without replacement using the query seed.",
        "- Flat exact search is used as the ground truth for top-k overlap@1, @5, and @10.",
        "- `top_k_overlap@K = |exact_top_K(query) intersect approximate_top_K(query)| / K`, averaged over all queries.",
        "- This is not biometric identification hit@K; top-k overlap can decrease when K grows.",
        "- Latency is measured as repeated single-query `index.search` calls after warmup.",
        "- Build time includes index construction and vector insertion; IVF-PQ build time also includes training.",
        "- Memory estimate is the serialized FAISS index size, not full process RSS.",
        "- The benchmark covers vector retrieval only; it excludes image decoding, face detection, embedding extraction, API, database, and UI latency.",
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
        "",
        "## Run Configuration",
        "",
        f"- Database size(s): `{sizes}`",
        f"- Embedding dimension: `{config['dim']}`",
        f"- Requested queries per size: `{config['n_queries']}`",
        f"- Seed: `{config['seed']}`",
        f"- Query seed: `{config['query_seed']}`",
        f"- Latency warmup runs: `{config['latency_warmup']}`",
        f"- Latency timed runs: `{config['latency_runs']}`",
        f"- K values: `{', '.join(str(k) for k in K_VALUES)}`",
        "",
        "## Results",
        "",
        "| Database size | Embedding dim | Queries | Seed | Method | Index parameters | Build time (s) | Memory estimate (MB) | p50 latency (ms) | p95 latency (ms) | p99 latency (ms) | top_k_overlap@1 | top_k_overlap@5 | top_k_overlap@10 |",
        "|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        latency = row["latency_ms"]
        top_k_overlap = row["top_k_overlap"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["n_vectors"]),
                    str(row["dim"]),
                    str(row["n_queries"]),
                    str(row["seed"]),
                    row["method"],
                    markdown_params(row["params"]),
                    format_value(row["build_time_s"]),
                    format_value(row["memory_mb"]),
                    format_value(latency["p50"]),
                    format_value(latency["p95"]),
                    format_value(latency["p99"]),
                    format_value(top_k_overlap["top_k_overlap@1"]),
                    format_value(top_k_overlap["top_k_overlap@5"]),
                    format_value(top_k_overlap["top_k_overlap@10"]),
                ]
            )
            + " |"
        )
        if row["status"] != "ok":
            lines.append(f"\nFailed row `{row['method']}`: `{row.get('error', '')}`\n")

    lines.append("")
    noted_rows = [row for row in rows if row.get("notes")]
    if noted_rows:
        lines.extend(["## Notes", ""])
        for row in noted_rows:
            lines.append(f"- `{row['n_vectors']}` / `{row['method']}`: {row['notes']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nSUMMARY - RETRIEVAL BENCHMARK")
    print(
        f"{'N':>8} {'Method':<18} {'O@1':>8} {'O@5':>8} {'O@10':>8} "
        f"{'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10} {'Build s':>10} {'MB':>10}"
    )
    print("-" * 110)
    for row in rows:
        latency = row["latency_ms"]
        top_k_overlap = row["top_k_overlap"]
        print(
            f"{row['n_vectors']:>8} {row['method']:<18} "
            f"{format_value(top_k_overlap['top_k_overlap@1'], 4):>8} "
            f"{format_value(top_k_overlap['top_k_overlap@5'], 4):>8} "
            f"{format_value(top_k_overlap['top_k_overlap@10'], 4):>8} "
            f"{format_value(latency['p50'], 4):>10} "
            f"{format_value(latency['p95'], 4):>10} "
            f"{format_value(latency['p99'], 4):>10} "
            f"{format_value(row['build_time_s'], 4):>10} "
            f"{format_value(row['memory_mb'], 4):>10}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FAISS retrieval methods")
    parser.add_argument(
        "--source-index",
        default=None,
        help="Path to existing FAISS index to reconstruct vectors from",
    )
    parser.add_argument(
        "--n-vectors",
        type=positive_int,
        default=9240,
        help="Number of generated vectors when --sizes is not provided",
    )
    parser.add_argument(
        "--sizes",
        type=positive_int,
        nargs="+",
        default=None,
        help="Generated-vector database sizes to benchmark",
    )
    parser.add_argument("--dim", type=positive_int, default=512, help="Vector dimensionality")
    parser.add_argument("--n-queries", type=positive_int, default=500, help="Queries per size")
    parser.add_argument("--seed", type=int, default=42, help="Generated vector seed")
    parser.add_argument("--query-seed", type=int, default=123, help="Query sampling seed")
    parser.add_argument("--latency-warmup", type=positive_int, default=50)
    parser.add_argument("--latency-runs", type=positive_int, default=500)
    parser.add_argument("--faiss-threads", type=positive_int, default=None)
    parser.add_argument("--output", default="training/outputs/retrieval_benchmark.json")
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--hnsw-m", type=positive_int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=positive_int, default=200)
    parser.add_argument("--hnsw-ef-search", type=positive_int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--ivfpq-nlist", type=positive_int, default=100)
    parser.add_argument("--ivfpq-m", type=positive_int, default=16)
    parser.add_argument("--ivfpq-nbits", type=positive_int, default=8)
    parser.add_argument("--ivfpq-nprobe", type=positive_int, nargs="+", default=[4, 8, 16, 32])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.faiss_threads is not None:
        faiss.omp_set_num_threads(args.faiss_threads)

    output_path = Path(args.output)
    csv_output_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    markdown_output_path = (
        Path(args.markdown_output) if args.markdown_output else output_path.with_suffix(".md")
    )

    source_index_path = Path(args.source_index) if args.source_index else None
    if source_index_path is not None and args.sizes is not None:
        raise SystemExit("--sizes is only supported for generated-vector benchmarks")
    if source_index_path is not None and not source_index_path.exists():
        raise SystemExit(f"Source index not found: {source_index_path}")

    rows: list[dict[str, Any]] = []
    if source_index_path is not None:
        print(f"Loading vectors from {source_index_path} ...")
        vectors = load_vectors_from_index(source_index_path)
        rows.extend(benchmark_size(vectors, args, source_label=str(source_index_path)))
        sizes = [int(len(vectors))]
        dim = int(vectors.shape[1])
    else:
        sizes = args.sizes or [args.n_vectors]
        dim = args.dim
        for n_vectors in sizes:
            print(f"Generating {n_vectors} vectors, dim={dim}, seed={args.seed} ...")
            vectors = generate_vectors(n_vectors, dim, args.seed)
            rows.extend(benchmark_size(vectors, args, source_label="generated"))

    meta = {
        "environment": environment_snapshot(),
        "config": {
            "source_index": str(source_index_path) if source_index_path else None,
            "sizes": sizes,
            "dim": dim,
            "n_queries": args.n_queries,
            "seed": args.seed,
            "query_seed": args.query_seed,
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

    write_json(output_path, meta, rows)
    write_csv(csv_output_path, rows)
    write_markdown(markdown_output_path, meta, rows)
    print_summary(rows)
    print(f"\nJSON: {output_path}")
    print(f"CSV: {csv_output_path}")
    print(f"Markdown: {markdown_output_path}")


if __name__ == "__main__":
    main()
