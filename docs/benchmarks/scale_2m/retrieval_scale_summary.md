# 1M and 2M Synthetic Retrieval Scalability Results

## Benchmark Goal

This benchmark evaluates whether the vector retrieval layer can handle million-scale synthetic embedding collections on a laptop-class environment. It is focused on FAISS index scalability, latency, build time, and index size.

These results do not measure biometric verification accuracy, face recognition quality, FAR, FRR, EER, liveness detection, detector latency, extractor latency, API latency, or desktop UI latency.

## Environment and Methodology

- Hardware class: laptop-class environment with 24 GB RAM.
- Platform: Windows 11.
- CPU count reported by the benchmark script: 20.
- Embedding dimension: 512.
- Vectors: synthetic L2-normalized float32 embeddings.
- Query count: 50 for 1M and 2M runs.
- Seed: 42.
- Query seed: 123.
- Ground truth: blockwise exact top-K search over the memmap-backed vector set.
- Metric: `top_k_overlap@K`, not biometric identification hit@K.

Formula:

```text
top_k_overlap@K = |exact_top_K(query) intersect approximate_top_K(query)| / K
```

This metric measures agreement between an approximate FAISS index and exact vector retrieval. It is not biometric accuracy.

## Results

| Dataset size | Method/config | p50 ms | p95 ms | p99 ms | mean ms | Build time (s) | Index size (MB) | top_k_overlap@1 | top_k_overlap@5 | top_k_overlap@10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000,000 | HNSW M=32, efConstruction=150, efSearch=128 | 1.738500 | 2.304975 | 2.523790 | 1.740340 | 982.189322 | 2212.647921 | 0.860000 | 0.244000 | 0.168000 |
| 1,000,000 | IVF-PQ nlist=4096, m=32, nbits=8, nprobe=32 | 0.347850 | 0.392000 | 0.466494 | 0.353613 | 38.337446 | 46.678394 | 1.000000 | 0.208000 | 0.108000 |
| 2,000,000 | IVF-PQ nlist=4096, m=32, nbits=8, nprobe=32 | 1.096600 | 1.142425 | 1.159439 | 0.968592 | 54.552124 | 84.825367 | 1.000000 | 0.208000 | 0.104000 |

## Interpretation

The 1M HNSW run shows low millisecond latency, but it also has high build time and a larger index size. In this run, HNSW reached p99 latency of about 2.52 ms, while the serialized index estimate was about 2.2 GB and build time was about 982 seconds.

The 2M IVF-PQ run shows compact memory footprint and low latency. With 2,000,000 synthetic 512D embeddings, IVF-PQ produced an index size estimate of about 84.8 MB and p99 latency of about 1.16 ms.

IVF-PQ `top_k_overlap@5` and `top_k_overlap@10` are low in the current configuration. This means the current IVF-PQ configuration is memory-efficient and fast, but it requires additional tuning if higher agreement with exact retrieval is needed.

HNSW 2M was not run before defense because the 1M HNSW result already showed significant build time and index size. Running 2M HNSW would be a separate resource-heavy experiment.

These results evaluate vector retrieval scalability, not biometric verification accuracy.

## Limitations

- Synthetic embeddings do not contain identity labels.
- The benchmark does not report biometric hit@K.
- The benchmark does not report FAR, FRR, EER, or threshold calibration values.
- The benchmark does not include detector, embedding extractor, API, database, or desktop UI latency.
- Index size is estimated from serialized FAISS index files, not full process RSS.
- HNSW at 2M was not run in this benchmark set.

## Related Artifacts

- `docs/benchmarks/scale_2m/retrieval_scale_1m_results.json`
- `docs/benchmarks/scale_2m/retrieval_scale_1m_results.csv`
- `docs/benchmarks/scale_2m/retrieval_scale_1m_results.md`
- `docs/benchmarks/scale_2m/retrieval_scale_2m_ivfpq_results.json`
- `docs/benchmarks/scale_2m/retrieval_scale_2m_ivfpq_results.csv`
- `docs/benchmarks/scale_2m/retrieval_scale_2m_ivfpq_results.md`
