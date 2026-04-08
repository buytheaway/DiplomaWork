# Defense Q&A

## What is this project?

It is a fast biometric face search system built from a FastAPI backend, a PySide6 desktop client, FAISS indexes and a storage layer for metadata and embeddings.

## Why FAISS and not only a database?

SQL storage is good for records and metadata, but inefficient for fast nearest-neighbor search in high-dimensional embedding space.
That is why SQL stores records, while FAISS performs vector retrieval.

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
- embeddings are encrypted before being stored in the database;
- FAISS snapshots are encrypted on disk;
- audit logs record important actions;
- retention policy removes old audit records.

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
- production deployment would still need HTTPS, rate limiting and mature secret management.
