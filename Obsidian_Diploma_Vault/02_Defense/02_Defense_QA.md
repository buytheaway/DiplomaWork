# Defense Q&A

## What is this project?

It is a fast biometric face search system built from a FastAPI backend, a PySide6 desktop client, FAISS indexes and a storage layer for metadata and embeddings.

## Why FAISS and not only a database?

SQL storage is good for records and metadata, but inefficient for fast nearest-neighbor search in high-dimensional embedding space.
That is why SQL stores records, while FAISS performs vector retrieval.

## Why HNSW?

HNSW is a graph-based approximate nearest-neighbor index. It is useful when the gallery grows because it can reduce search latency while preserving high overlap with exact Flat results in the synthetic retrieval benchmark.

## What does top_k_overlap@K mean?

`top_k_overlap@K = |exact_top_K(query) ∩ approximate_top_K(query)| / K`.
It measures how much an approximate FAISS index agrees with exact Flat search on synthetic vectors.

## Why is top_k_overlap@K not biometric accuracy?

The synthetic benchmark has no identity labels. It measures vector retrieval behavior, not whether a person was correctly identified.

## Where are FAR, FRR, and EER?

The formulas and unit-tested helpers are implemented in `training/verification_metrics.py`.
Final numeric values require running `scripts/evaluate_verification_pairs.py` or another labeled pair evaluator on real labeled pairs.

## Why is the custom model experimental?

The custom PyTorch branch contains training and evaluation infrastructure, but the tracked documentation does not include valid final training logs or labeled verification results.
The stable MVP should be presented through the pretrained ONNX or InsightFace extractor path.

## What is stored in the system?

The system stores:

- `person`
- `embedding`
- `index snapshot`
- `audit log`

Raw images are not used as the default operational storage.

## Why is enroll limited to one face?

Because one record should correspond to one valid biometric sample.
This keeps the gallery cleaner and avoids ambiguous labels.

## Why does search allow multiple faces?

Because search analyzes a scene.
If several people are visible, the system can process them one by one and return per-face results.

## What is an embedding?

It is a numeric representation of a face, a fixed-length vector used for similarity search.

## Why compare mode?

To compare two pipelines on the same input by quality and latency.

## How are biometric data protected?

- API routes are protected by `API_KEY`;
- admin actions use `ADMIN_API_KEY`;
- API-key comparison is timing-safe;
- embeddings are encrypted before being stored in the database;
- FAISS snapshots are encrypted on disk;
- configurable in-memory rate limiting protects search, enroll, compare, rebuild, and delete routes;
- old FAISS snapshots are pruned by `INDEX_SNAPSHOT_RETENTION`;
- audit logs record important actions;
- retention policy removes old audit records.

## Is the system ready for enterprise deployment?

No. It is a local MVP and diploma prototype with basic hardening.
Distributed deployments would need HTTPS termination, secret rotation, Redis or gateway-level rate limiting, fuller RBAC, operational monitoring, and a hard purge policy.

## Where are vectors stored?

Vectors are stored in the `embeddings` table in the `vector` column.
In PostgreSQL that column is `BYTEA`.
It no longer stores raw float bytes directly; it stores an encrypted payload.

## What is already implemented?

- search
- enroll
- compare
- multi-face search
- webcam mode
- database view
- logs
- index rebuild
- basic security hardening

## What are the limitations?

- part of the custom branch is still research-oriented;
- live webcam is near real-time, not a heavy industrial stream;
- enterprise deployment would still need HTTPS, distributed rate limiting, mature secret management, fuller RBAC, and hard purge support.
