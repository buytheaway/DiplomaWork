"""Benchmark FAISS retrieval methods: Flat vs HNSW vs IVF-PQ.

Measures recall@k, latency, memory, and build time for each index type
on a given set of embeddings.  Results are saved to a JSON file and
printed as a table suitable for the diploma.

Usage::

    # Use embeddings from a FAISS index or generate random ones:
    python scripts/benchmark_retrieval.py

    # Use an existing FAISS index:
    python scripts/benchmark_retrieval.py --source-index backend/data/index/test3.faiss

    # Specify parameters:
    python scripts/benchmark_retrieval.py --n-vectors 10000 --dim 512 --n-queries 500
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_flat(vectors: np.ndarray) -> faiss.Index:
    """Build a flat (exact) inner-product index."""
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    return index


def build_hnsw(
    vectors: np.ndarray, M: int = 32, ef_construction: int = 200
) -> faiss.Index:
    """Build an HNSW index."""
    d = vectors.shape[1]
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def build_ivfpq(
    vectors: np.ndarray, nlist: int = 100, M_pq: int = 16, nbits: int = 8
) -> faiss.Index:
    """Build an IVF-PQ index."""
    d = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, M_pq, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    return index


def measure_recall(
    index: faiss.Index,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k_values: list[int],
) -> dict[str, float]:
    """Compute recall@k for given queries against ground truth (from Flat index)."""
    max_k = max(k_values)
    _, I = index.search(queries, max_k)

    results = {}
    for k in k_values:
        hits = 0
        for q_idx in range(len(queries)):
            gt_set = set(ground_truth[q_idx, :k].tolist())
            pred_set = set(I[q_idx, :k].tolist())
            hits += len(gt_set & pred_set) / k
        results[f"recall@{k}"] = round(hits / len(queries), 4)
    return results


def measure_latency(
    index: faiss.Index,
    queries: np.ndarray,
    k: int = 10,
    n_warmup: int = 50,
    n_runs: int = 500,
) -> float:
    """Measure average single-query latency in ms."""
    # Warmup
    for i in range(min(n_warmup, len(queries))):
        index.search(queries[i : i + 1], k)

    # Timed runs
    n_runs = min(n_runs, len(queries))
    total = 0.0
    for i in range(n_runs):
        start = time.perf_counter()
        index.search(queries[i : i + 1], k)
        total += time.perf_counter() - start

    return round(total / n_runs * 1000, 4)  # ms


def measure_memory_mb(index: faiss.Index) -> float:
    """Rough memory estimate by serializing to bytes."""
    writer = faiss.VectorIOWriter()
    faiss.write_index(index, writer)
    return round(len(faiss.vector_to_array(writer.data)) / 1024 / 1024, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FAISS retrieval methods")
    parser.add_argument("--source-index", default=None, help="Path to existing FAISS index to extract vectors from")
    parser.add_argument("--n-vectors", type=positive_int, default=9240, help="Number of vectors (if generating random)")
    parser.add_argument("--dim", type=positive_int, default=512, help="Vector dimensionality")
    parser.add_argument("--n-queries", type=positive_int, default=500, help="Number of test queries")
    parser.add_argument("--output", default="training/outputs/retrieval_benchmark.json")
    # HNSW params
    parser.add_argument("--hnsw-m", type=positive_int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=positive_int, default=200)
    parser.add_argument("--hnsw-ef-search", type=positive_int, nargs="+", default=[32, 64, 128])
    # IVF-PQ params
    parser.add_argument("--ivfpq-nlist", type=positive_int, default=100)
    parser.add_argument("--ivfpq-m", type=positive_int, default=16)
    parser.add_argument("--ivfpq-nbits", type=positive_int, default=8)
    parser.add_argument("--ivfpq-nprobe", type=positive_int, nargs="+", default=[4, 8, 16, 32])
    args = parser.parse_args()

    # --- Load or generate vectors ---
    if args.source_index and Path(args.source_index).exists():
        print(f"Loading vectors from {args.source_index} ...")
        src_index = faiss.read_index(args.source_index)
        n = src_index.ntotal
        d = src_index.d
        vectors = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            vectors[i] = src_index.reconstruct(i)
        faiss.normalize_L2(vectors)
        print(f"  Loaded {n} vectors, dim={d}")
    else:
        n, d = args.n_vectors, args.dim
        print(f"Generating {n} random vectors, dim={d} ...")
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, d).astype(np.float32)
        faiss.normalize_L2(vectors)

    # --- Prepare queries ---
    n_queries = min(args.n_queries, n)
    query_indices = np.random.RandomState(123).choice(n, size=n_queries, replace=False)
    queries = vectors[query_indices].copy()

    # --- Ground truth from Flat ---
    print("\nBuilding ground truth (Flat exact search)...")
    flat_index = build_flat(vectors)
    k_values = [1, 5, 10]
    max_k = max(k_values)
    _, gt_I = flat_index.search(queries, max_k)

    all_results = {}

    # --- Benchmark Flat ---
    print("\n" + "=" * 60)
    print("Benchmarking: Flat (exact)")
    print("=" * 60)
    start = time.perf_counter()
    flat = build_flat(vectors)
    build_time = time.perf_counter() - start

    recall = measure_recall(flat, queries, gt_I, k_values)
    latency = measure_latency(flat, queries, k=10)
    memory = measure_memory_mb(flat)

    all_results["flat"] = {
        "recall": recall,
        "latency_ms": latency,
        "memory_mb": memory,
        "build_time_s": round(build_time, 4),
    }
    print(f"  Recall: {recall}")
    print(f"  Latency: {latency:.4f} ms/query")
    print(f"  Memory: {memory:.2f} MB")
    print(f"  Build time: {build_time:.4f} s")

    # --- Benchmark HNSW ---
    for ef_search in args.hnsw_ef_search:
        name = f"hnsw_ef{ef_search}"
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: HNSW (M={args.hnsw_m}, efSearch={ef_search})")
        print("=" * 60)

        start = time.perf_counter()
        hnsw = build_hnsw(vectors, M=args.hnsw_m, ef_construction=args.hnsw_ef_construction)
        build_time = time.perf_counter() - start

        hnsw.hnsw.efSearch = ef_search
        recall = measure_recall(hnsw, queries, gt_I, k_values)
        latency = measure_latency(hnsw, queries, k=10)
        memory = measure_memory_mb(hnsw)

        all_results[name] = {
            "params": {"M": args.hnsw_m, "efConstruction": args.hnsw_ef_construction, "efSearch": ef_search},
            "recall": recall,
            "latency_ms": latency,
            "memory_mb": memory,
            "build_time_s": round(build_time, 4),
        }
        print(f"  Recall: {recall}")
        print(f"  Latency: {latency:.4f} ms/query")
        print(f"  Memory: {memory:.2f} MB")
        print(f"  Build time: {build_time:.4f} s")

    # --- Benchmark IVF-PQ ---
    for nprobe in args.ivfpq_nprobe:
        name = f"ivfpq_nprobe{nprobe}"
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: IVF-PQ (nlist={args.ivfpq_nlist}, M_pq={args.ivfpq_m}, nprobe={nprobe})")
        print("=" * 60)

        start = time.perf_counter()
        ivfpq = build_ivfpq(
            vectors,
            nlist=args.ivfpq_nlist,
            M_pq=args.ivfpq_m,
            nbits=args.ivfpq_nbits,
        )
        build_time = time.perf_counter() - start

        ivfpq.nprobe = nprobe
        recall = measure_recall(ivfpq, queries, gt_I, k_values)
        latency = measure_latency(ivfpq, queries, k=10)
        memory = measure_memory_mb(ivfpq)

        all_results[name] = {
            "params": {"nlist": args.ivfpq_nlist, "M_pq": args.ivfpq_m, "nbits": args.ivfpq_nbits, "nprobe": nprobe},
            "recall": recall,
            "latency_ms": latency,
            "memory_mb": memory,
            "build_time_s": round(build_time, 4),
        }
        print(f"  Recall: {recall}")
        print(f"  Latency: {latency:.4f} ms/query")
        print(f"  Memory: {memory:.2f} MB")
        print(f"  Build time: {build_time:.4f} s")

    # --- Summary table ---
    print(f"\n\n{'=' * 80}")
    print("SUMMARY — RETRIEVAL BENCHMARK")
    print(f"{'=' * 80}")
    print(f"{'Method':<25} {'Recall@1':>10} {'Recall@5':>10} {'Recall@10':>10} {'Latency':>10} {'Memory':>10}")
    print("-" * 80)
    for method, data in all_results.items():
        r = data["recall"]
        print(
            f"{method:<25} {r.get('recall@1', 0):>10.4f} {r.get('recall@5', 0):>10.4f} "
            f"{r.get('recall@10', 0):>10.4f} {data['latency_ms']:>8.4f}ms {data['memory_mb']:>8.2f}MB"
        )
    print(f"{'=' * 80}")

    # --- Save ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_results["_meta"] = {
        "n_vectors": n,
        "dim": d,
        "n_queries": n_queries,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
