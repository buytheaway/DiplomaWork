# Retrieval Benchmark Results

Generated at: `2026-05-03T13:33:46.885127+00:00`

## Methodology

- Synthetic embeddings are generated with NumPy RandomState unless a source FAISS index is provided.
- All vectors are L2-normalized and searched with inner-product similarity.
- Query vectors are sampled from the benchmark database without replacement using the query seed.
- Flat exact search is used as the ground truth for top-k overlap@1, @5, and @10.
- `top_k_overlap@K = |exact_top_K(query) intersect approximate_top_K(query)| / K`, averaged over all queries.
- This is not biometric identification hit@K; top-k overlap can decrease when K grows.
- Latency is measured as repeated single-query `index.search` calls after warmup.
- Build time includes index construction and vector insertion; IVF-PQ build time also includes training.
- Memory estimate is the serialized FAISS index size, not full process RSS.
- The benchmark covers vector retrieval only; it excludes image decoding, face detection, embedding extraction, API, database, and UI latency.

## Metric definition

- `top_k_overlap@K = |exact_top_K(query) intersect approximate_top_K(query)| / K`, averaged over all queries.
- This metric compares the approximate index output with the exact Flat top-K neighbors.
- It is not biometric identification hit@K, because this synthetic benchmark has no identity labels.
- `top_k_overlap@1` may be higher than `top_k_overlap@5` or `top_k_overlap@10` when the approximate index recovers the nearest exact neighbor but misses part of the wider exact top-K neighborhood.

## Interpretation

- Flat is the exact baseline used to compute the reference nearest-neighbor lists.
- On the 10,000-vector run, HNSW keeps high top-K overlap while reducing p50/p95/p99 search latency compared with Flat.
- On very small datasets, Flat is faster because approximate-index overhead is not justified by the database size.
- IVF-PQ uses less serialized index memory in this configuration, but the current small-scale parameters produce very poor top-K overlap on the synthetic vectors. In particular, `top_k_overlap@1` is only `0.130000` at 1,000 vectors and `0.030000` at 10,000 vectors, so this IVF-PQ setting must not be presented as a working quality-preserving index.
- For the current PR2 small-scale results, HNSW is the appropriate approximate index to discuss as quality-preserving. IVF-PQ should be framed as a compressed-index experiment that requires tuning and/or reranking before use in a recognition-critical path.
- This synthetic benchmark evaluates vector retrieval behavior, not biometric recognition accuracy.

## Limitations

- The benchmark uses synthetic L2-normalized embeddings.
- The benchmark has no identity labels.
- The benchmark does not compute biometric identification hit@K.
- The benchmark does not compute FAR, FRR, or EER.
- A 100,000-vector run was not executed in this PR.

## Hardware/Environment

- Platform: `Windows-11-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel`
- Machine: `AMD64`
- CPU count: `20`
- Python: `3.12.10`
- NumPy: `2.4.2`
- FAISS: `1.13.2`
- FAISS threads: `20`

## Run Configuration

- Database size(s): `100, 1000, 10000`
- Embedding dimension: `512`
- Requested queries per size: `100`
- Seed: `42`
- Query seed: `123`
- Latency warmup runs: `10`
- Latency timed runs: `100`
- K values: `1, 5, 10`

## Results

| Database size | Embedding dim | Queries | Seed | Method | Index parameters | Build time (s) | Memory estimate (MB) | p50 latency (ms) | p95 latency (ms) | p99 latency (ms) | top_k_overlap@1 | top_k_overlap@5 | top_k_overlap@10 |
|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 512 | 100 | 42 | flat | `{"metric":"inner_product","normalized_vectors":true}` | 0.000081 | 0.195355 | 0.004500 | 0.008610 | 0.021691 | 1.000000 | 1.000000 | 1.000000 |
| 100 | 512 | 100 | 42 | hnsw_ef64 | `{"M":32,"efConstruction":200,"efSearch":64,"metric":"inner_product","normalized_vectors":true}` | 0.001760 | 0.221331 | 0.022700 | 0.024515 | 0.034254 | 1.000000 | 1.000000 | 1.000000 |
| 100 | 512 | 100 | 42 | ivfpq_nprobe8 | `{"M_pq":16,"metric":"inner_product","nbits":4,"nlist":16,"normalized_vectors":true,"nprobe":8}` | 0.019342 | 0.064320 | 0.008500 | 0.009210 | 0.009929 | 1.000000 | 0.290000 | 0.401000 |
| 1000 | 512 | 100 | 42 | flat | `{"metric":"inner_product","normalized_vectors":true}` | 0.000552 | 1.953168 | 0.037450 | 0.052320 | 0.059548 | 1.000000 | 1.000000 | 1.000000 |
| 1000 | 512 | 100 | 42 | hnsw_ef64 | `{"M":32,"efConstruction":200,"efSearch":64,"metric":"inner_product","normalized_vectors":true}` | 0.013336 | 2.212221 | 0.061900 | 0.070200 | 0.072136 | 1.000000 | 0.994000 | 0.997000 |
| 1000 | 512 | 100 | 42 | ivfpq_nprobe8 | `{"M_pq":16,"metric":"inner_product","nbits":4,"nlist":16,"normalized_vectors":true,"nprobe":8}` | 0.070546 | 0.078053 | 0.019800 | 0.020505 | 0.021358 | 0.130000 | 0.056000 | 0.052000 |
| 10000 | 512 | 100 | 42 | flat | `{"metric":"inner_product","normalized_vectors":true}` | 0.005819 | 19.531293 | 0.616750 | 0.976610 | 1.045482 | 1.000000 | 1.000000 | 1.000000 |
| 10000 | 512 | 100 | 42 | hnsw_ef64 | `{"M":32,"efConstruction":200,"efSearch":64,"metric":"inner_product","normalized_vectors":true}` | 1.786023 | 22.123201 | 0.333450 | 0.460055 | 0.472427 | 1.000000 | 0.994000 | 0.987000 |
| 10000 | 512 | 100 | 42 | ivfpq_nprobe8 | `{"M_pq":16,"metric":"inner_product","nbits":4,"nlist":16,"normalized_vectors":true,"nprobe":8}` | 0.174341 | 0.215382 | 0.131150 | 0.148080 | 0.167981 | 0.030000 | 0.012000 | 0.009000 |

## Notes

- `100` / `ivfpq_nprobe8`: FAISS clustering warning: this IVF-PQ configuration has fewer training vectors than the usual recommendation (100 < 624).
