# Final Defense Q&A

## Why FAISS?

The database is responsible for metadata, identifiers, labels, audit records,
and persistence. FAISS is responsible for nearest-neighbor retrieval in
high-dimensional embedding space. This separation makes the search layer faster
and easier to evaluate.

## Why HNSW?

HNSW is a graph-based approximate nearest-neighbor index. In the synthetic
benchmark, HNSW preserves high top-k overlap on the 10,000-vector run while
reducing search latency compared with exact Flat search.

## Why IVF-PQ?

IVF-PQ combines an inverted file with Product Quantization. It is useful for
studying memory-saving retrieval configurations. In the current benchmark, its
serialized index size is lower, but its top-k overlap is also low, so it should
be presented as an experimental memory/quality trade-off.

## What Is top_k_overlap@K?

`top_k_overlap@K = |exact_top_K(query) intersection approximate_top_K(query)| / K`.

It measures how much an approximate index agrees with exact Flat search on
synthetic vectors.

## Why Is top_k_overlap@K Not Biometric Accuracy?

The synthetic retrieval benchmark has no identity labels. It measures vector
retrieval behavior after embeddings already exist. It does not measure whether
the correct person was identified.

## Where Are FAR, FRR, And EER?

The formulas and unit-tested helpers are implemented in
`training/verification_metrics.py`. The stable extractor pair evaluator is
`scripts/evaluate_verification_pairs.py`.

## Why Are There No Real FAR/FRR/EER Numbers Yet?

Real FAR, FRR, EER, TAR@FAR, and threshold values require labeled positive and
negative biometric pairs. The tracked documentation states that LFW-style
stable extractor evaluation has not been run yet.

## Why Is The Custom Model Experimental?

The custom PyTorch branch contains training and evaluation infrastructure, but
the tracked artifacts do not include final training evidence and labeled
verification results. The stable MVP should be presented through the pretrained
ONNX or InsightFace extractor path.

## How Are Embeddings Protected?

New embeddings are encrypted before being stored in the database. FAISS
snapshots and their `.map.json` sidecars are encrypted on disk. API-key
comparison is timing-safe, rate-limit buckets do not log raw API keys, and
audit logs record important actions.

## Is The System Ready For Production Deployment?

No. It is a local diploma MVP with basic hardening. A real deployment would
need HTTPS operations, secret rotation, distributed rate limiting, fuller RBAC,
monitoring, backup policy, liveness or spoofing controls, and hard purge
support.

## What Is The Main Contribution?

The main contribution is an end-to-end biometric face search system: modular
embedding extractors, FAISS vector indexing, relational metadata storage,
desktop operator workflow, synthetic retrieval benchmarking, verification
metric methodology, and MVP security/privacy controls.
