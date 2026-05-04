# Final Demo Script

Target duration: 5-7 minutes.

## 0. Preparation

Use demo or synthetic data only. Do not use real biometric data with example
configuration values.

Start backend:

```powershell
cd backend
.\.venv\Scripts\activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start desktop client in a second terminal:

```powershell
cd desktop
.\.venv\Scripts\activate
python -m app.main
```

## 1. Dashboard

Open the Dashboard tab.

Say:

"This is the operator entry point. It shows backend availability, active
pipelines, indexed vectors, and recent activity. The system is designed as an
operator-assisted biometric search tool, not just as a backend API."

## 2. Index Stats

Open Logs or index tools and show index statistics.

Say:

"Metadata is stored in the relational database, while nearest-neighbor vector
retrieval is handled by FAISS. Each pipeline has its own index and snapshot
state."

## 3. Enroll One Image

Open Face Search, switch to Enroll, select one clear demo image, enter a label,
and enroll it.

Say:

"Enrollment is stricter than search. It expects one valid face because one
record should correspond to one biometric sample. Raw images are not stored by
default; the long-term artifact is an encrypted embedding."

## 4. Search Image

Switch to Search, select a query image, and run search.

Show:

- top-k candidates;
- similarity scores;
- threshold decision;
- detected faces;
- latency information if visible.

Say:

"The query image is converted into an embedding. FAISS retrieves nearest
candidate vectors, and the backend enriches matches with metadata from the
database."

## 5. Compare Mode

Run Compare mode on the same image.

Say:

"Compare mode processes the same input through available pipelines. This is
useful for controlled comparison, but current documentation does not claim
validated custom-model biometric accuracy."

## 6. Database And Logs

Open Database and Logs.

Say:

"The database view makes enrolled records inspectable. Logs and audit records
support operational review. Delete is a soft-delete workflow; hard purge is
future work."

## 7. Benchmark Table

Open `docs/benchmarks/retrieval_benchmark_pr2.md` or the diploma benchmark
section.

Show:

- Flat exact baseline;
- HNSW;
- IVF-PQ;
- p50/p95/p99 latency;
- build time;
- memory estimate;
- `top_k_overlap@1`, `top_k_overlap@5`, `top_k_overlap@10`.

Say:

"This is a synthetic vector retrieval benchmark. It evaluates FAISS retrieval
behavior, not biometric recognition accuracy. The metric is
`top_k_overlap@K`, the overlap with exact Flat top-K neighbors."

## 8. Security And Privacy Summary

Show README or diploma security section.

Say:

"The MVP includes timing-safe API-key comparison, encrypted embeddings,
encrypted FAISS snapshots, configurable in-memory rate limiting, audit logs,
and snapshot retention. It is still a diploma MVP: distributed rate limiting,
full RBAC, hard purge, HTTPS operations, and liveness detection remain future
work."
