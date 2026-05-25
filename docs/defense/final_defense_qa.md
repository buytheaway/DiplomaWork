# Final Defense Q&A

## Why FAISS?

The database is responsible for metadata, identifiers, labels, audit records,
and persistence. FAISS is responsible for nearest-neighbor retrieval in
high-dimensional embedding space. This separation makes the search layer faster
and easier to evaluate.

## Why HNSW?

HNSW is a graph-based approximate nearest-neighbor index. In this project it is
used as the main practical index type for the real-image scale database because
it gives low-latency nearest-neighbor search while keeping the retrieval path
simple enough for the final demo.

## Why IVF-PQ?

IVF-PQ combines an inverted file with Product Quantization. It is useful for
studying memory-saving retrieval configurations. In the current benchmark, its
serialized index size is lower, but its top-k overlap is also low, so it should
be presented as an experimental memory/quality trade-off.

## What Is top_k_overlap@K?

`top_k_overlap@K = |exact_top_K(query) intersection approximate_top_K(query)| / K`.

It measures how much an approximate index agrees with exact Flat search on the
same vector collection.

## Why Is Retrieval Latency Not Biometric Accuracy?

Retrieval latency measures how fast the index returns nearest vectors after an
embedding already exists. Biometric accuracy requires labeled positive and
negative face pairs or a held-out identification protocol.

## Where Are FAR, FRR, And EER?

The formulas and unit-tested helpers are implemented in
`training/verification_metrics.py`. The final LFW evaluator is
`scripts/evaluate_lfw_verification.py`, and the tracked result summary is
`docs/benchmarks/lfw_biometric_verification_results.md`.

## What Do The LFW Results Show?

The final custom `torch_insightface_iresnet100` pipeline has real LFW
verification metrics: EER `0.015000`, best accuracy `0.990500`, and
TAR@FAR=0.01 `0.984667` on 6000/6000 valid pairs. The pretrained
ONNX/InsightFace baseline is the external reference: EER `0.027852`, best
accuracy `0.984556`, and TAR@FAR=0.01 `0.971141`.

## Why Is The Custom Model Experimental?

The custom PyTorch branch is implemented and measured with real LFW biometric
metrics. It should still be presented with boundaries: the result does not prove
liveness resistance, universal deployment accuracy, or compatibility with old
`torch_ir50` embeddings.

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
desktop operator workflow, real-image retrieval benchmarking, verification
metric methodology, and MVP security/privacy controls.
