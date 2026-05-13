# Large-Scale Synthetic Retrieval Benchmark Results

Generated at: `2026-05-13T21:05:47.180500+00:00`

## Methodology

- Synthetic vectors are generated as L2-normalized float32 embeddings.
- Base vectors are stored in an ignored `np.memmap` file under the configured temporary directory.
- Exact top-K ground truth is computed blockwise with inner-product similarity.
- FAISS methods are built and benchmarked one at a time to reduce peak memory usage.
- Index size is measured by writing a temporary FAISS index file and measuring file size.
- Temporary index files are deleted unless `--keep-index-files` is passed.

## Metric Definition

`top_k_overlap@K = |exact_top_K(query) intersect approximate_top_K(query)| / K`

This is not biometric identification hit@K and not biometric accuracy. The benchmark uses synthetic vectors with no identity labels, so it evaluates vector retrieval behavior only.

## Hardware/Environment

- Platform: `Windows-11-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel`
- Machine: `AMD64`
- CPU count: `20`
- Python: `3.12.10`
- NumPy: `2.4.2`
- FAISS: `1.13.2`
- FAISS threads: `20`
- Full process RSS is not measured; memory estimate is based on temporary FAISS index file size.

## Run Configuration

- Sizes: `2000000`
- Embedding dimension: `512`
- Queries: `50`
- Seed: `42`
- Query seed: `123`
- Batch size: `50000`
- Train size: `200000`
- Methods: `ivfpq`
- K values: `1, 5, 10`

## Results

| N | Dim | Queries | Method | Status | Build time (s) | Mean ms | p50 ms | p95 ms | p99 ms | Memory MB | Index file MB | top_k_overlap@1 | top_k_overlap@5 | top_k_overlap@10 | Notes |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2000000 | 512 | 50 | ivfpq | ok | 54.552124 | 0.968592 | 1.096600 | 1.142425 | 1.159439 | 84.825367 | 84.825367 | 1.000000 | 0.208000 | 0.104000 |  |

## Limitations

- The benchmark uses synthetic L2-normalized vectors, not real biometric identities.
- It does not measure FAR, FRR, EER, liveness resistance, API latency, detector latency, or UI latency.
- `top_k_overlap@K` measures agreement with exact blockwise top-K retrieval, not recognition accuracy.
- Results should only be reported after the script is actually run on the target hardware.