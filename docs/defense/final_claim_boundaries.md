# Final Claim Boundaries

| Claim | Allowed? | Safe wording | Evidence | Forbidden wording |
|---|---:|---|---|---|
| Fast vector search is implemented | Yes | FAISS-based vector retrieval is implemented in the backend. | `backend/app/services/index/*`, route tests | The whole biometric system is proven fast at any scale. |
| Flat/HNSW/IVF-PQ are compared | Yes | The synthetic benchmark compares Flat, HNSW, and IVF-PQ retrieval behavior. | `scripts/benchmark_retrieval.py`, `docs/benchmarks/retrieval_benchmark_pr2.md` | IVF-PQ is always better than Flat or HNSW. |
| `top_k_overlap@K` is reported | Yes | `top_k_overlap@K` measures overlap with exact Flat top-K neighbors. | PR2 benchmark artifacts | It is biometric identification hit@K. |
| Synthetic retrieval benchmark shows biometric accuracy | No | The benchmark evaluates vector retrieval behavior only. | Benchmark methodology and limitations | The benchmark proves recognition quality. |
| FAR/FRR/EER methodology exists | Yes | FAR, FRR, EER, TAR@FAR helpers and tests exist. | `training/verification_metrics.py`, tests | Final FAR/FRR/EER values were measured. |
| Stable extractor evaluator exists | Yes | The evaluator can run ONNX, InsightFace, Torch, or Dummy backends on labeled pairs. | `scripts/evaluate_verification_pairs.py`, tests | Stable ONNX accuracy is already confirmed. |
| Custom Torch runtime pipeline exists | Yes | The custom Torch pipeline is implemented as a runtime pipeline for a deployed IR-50 model bundle. In the prepared demo environment, pretrained ONNX and custom Torch indexes can both be available. | backend pipeline registry, import script, runtime stats | The custom model is better than the pretrained model, or final system-level biometric accuracy is confirmed. |
| Custom PyTorch branch quality is validated | No | The custom PyTorch branch is experimental and needs labeled evaluation. | Training code and readiness docs | The custom branch has completed quality validation or outperforms the pretrained path. |
| Raw images are not stored by default | Yes | Runtime requests use image bytes for extraction and do not store raw images by default. | routes and storage code | No biometric privacy risk remains. |
| Embeddings and snapshots are encrypted | Yes, for newly stored artifacts | New embeddings and FAISS snapshots are encrypted when encryption keys are configured. | `crypto.py`, repositories, index manager, security tests | Encryption alone makes the system compliant. |
| API-key comparison is timing-safe | Yes | API keys are compared with timing-safe comparison. | `backend/app/security/auth.py`, tests | Authentication is enterprise-complete. |
| Rate limiting exists | Yes | Configurable per-process in-memory rate limiting protects MVP search/enroll/admin flows. | `backend/app/security/rate_limit.py`, tests | Full DDoS protection is implemented. |
| Snapshot retention exists | Yes | Old snapshots are pruned according to `INDEX_SNAPSHOT_RETENTION`. | `index_manager.py`, retention tests | Snapshot retention is a legal deletion policy. |
| Hard purge exists | No | Soft delete exists; hard purge is future work. | persons route and tests | Irreversible deletion is implemented. |
| System is deployment-complete | No | The project is a diploma MVP with basic hardening. | README and security docs | Enterprise deployment requirements are fully satisfied. |
