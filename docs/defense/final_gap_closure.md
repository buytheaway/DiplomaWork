# Final Gap Closure

This note maps the remaining supervisor comments to concrete project evidence
and safe wording for the final diploma version.

## 1. Real Biometric Metrics

The project now reports real labeled-pair biometric metrics on LFW:

| Pipeline | EER | Best accuracy | TAR@FAR=0.01 | Role |
|---|---:|---:|---:|---|
| Final custom `torch_insightface_iresnet100` pipeline | 0.015000 | 0.990500 | 0.984667 | Final custom runtime |
| Pretrained ONNX/InsightFace baseline | 0.027852 | 0.984556 | 0.971141 | External baseline/reference |

Safe claim: the final custom runtime was evaluated with FAR/FRR/EER-style
metrics and compared with a pretrained baseline.

Unsafe claim: the final custom model is guaranteed to outperform every external
model in every operating condition.

## 2. IVF-PQ Interpretation

Safe claim: IVF-PQ was evaluated as a compressed memory-saving experimental
index type. For the final runtime demo, HNSW is the practical default because
it supports low-latency retrieval on the real-image scale database without
changing the biometric verification protocol.

Recommended demo/research framing:

- HNSW is the primary final-demo index for the custom runtime.
- IVF-PQ is relevant for memory/scalability experiments, but requires parameter
  tuning and/or reranking before being used as a high-quality retrieval path.

## 3. Final Dataset Definition

Final dataset roles:

| Dataset/source | Role | Used for |
|---|---|---|
| `datasets/celeba_faces/train` | Main real-face training dataset | Custom Torch fine-tuning and candidate experiments |
| `datasets/celeba_faces/val` | Validation split | Training sanity checks and dataset validation |
| `handoff_lfw_eval/lfw` | Evaluation-only dataset | FAR/FRR/EER/TAR@FAR verification |
| Scale PostgreSQL database | Real-image-derived retrieval database | Database UI pagination, indexed vector counts, runtime smoke checks |

LFW must be described as evaluation-only in the final diploma. It should not be
presented as the final training dataset.

## 4. Data Preparation Pipeline

The data pipeline is:

1. Organize real training data as identity folders:
   `datasets/celeba_faces/train/<identity>/*.jpg` and
   `datasets/celeba_faces/val/<identity>/*.jpg`.
2. Validate image readability and identity folder structure.
3. Detect and align faces to ArcFace-style `112x112` crops when running aligned
   training experiments.
4. Normalize image tensors with RGB order and `[-1, 1]` scaling for the custom
   Torch training and candidate evaluation pipeline.
5. Train or fine-tune the custom Torch backbone with ArcFace-style
   classification and, in experiments, optional teacher embedding distillation.
6. Evaluate checkpoints on LFW labeled pairs using dot-product similarity of
   L2-normalized embeddings.
7. For runtime deployment, rebuild all custom stored embeddings and FAISS
   indexes after any checkpoint or preprocessing change.

## 5. Scientific And Engineering Contribution Boundary

Implemented by this project:

- FastAPI backend and API contracts for enroll/search/database/index operations.
- PySide6 desktop operator client.
- Pipeline registry for pretrained and custom embedding backends.
- Custom Torch runtime integration and evaluation scripts, including the final
  `torch_insightface_iresnet100` candidate.
- FAISS index management, snapshot handling, and real-image retrieval benchmarks.
- PostgreSQL/SQLite metadata storage and paginated database views.
- Security hardening for API keys, encryption of stored embeddings/snapshots,
  rate limiting, and audit-friendly operations.
- Real LFW verification evaluator and comparison reporting.

External methods/libraries used by the project:

- ArcFace/InsightFace-style pretrained baseline and face alignment concepts.
- SCRFD/ONNX detector and ONNX Runtime.
- FAISS Flat, HNSW, and IVF-PQ algorithms.
- PyTorch, FastAPI, SQLAlchemy, PySide6.

Safe novelty wording: the contribution is an integrated, reproducible biometric
face-search system and evaluation framework, not a newly invented face
recognition loss or a new ANN algorithm.
