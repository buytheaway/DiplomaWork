# Fast Biometric Face Search

Production‑quality diploma prototype for face‑based biometric search.

**Stack:** FastAPI · PostgreSQL / SQLite · FAISS · PySide6 desktop client.

**Key design principle:** the ML model is a *plugin*. The system boots and works
with `EMBEDDING_BACKEND=dummy` (zero ML dependencies). Swap to a real model
(InsightFace / PyTorch / ONNX) by changing one env var + installing the lib.

---

## Quick start (Docker — recommended)

```powershell
# 1. Clone & enter repo
Set-Location "C:\Users\mukha\OneDrive\Documents\GitHub\DiplomaWork"

# 2. Create .env (defaults to EMBEDDING_BACKEND=dummy)
if (-not (Test-Path .env)) { Copy-Item .env.example .env }

# 3. Start services
docker compose up --build
```

Health check:

```powershell
Invoke-WebRequest http://localhost:8000/v1/health
# or: curl http://127.0.0.1:8000/v1/health
```

Swagger docs: <http://localhost:8000/docs>

### With a real ML model (Docker)

```powershell
# Rebuild with ML deps
docker compose build --build-arg INSTALL_ML=true
# Update .env:  EMBEDDING_BACKEND=insightface  (or torch / onnx)
docker compose up
```

---

## Run on Windows (local, no Docker)

### Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Optional ML: pip install -r requirements-ml.txt

# Create .env in repo root (SQLite for local dev):
Copy-Item ..\.env.local.example ..\.env -Force
# Or just use dummy backend:
Copy-Item ..\.env.example ..\.env -Force

alembic upgrade head
uvicorn app.main:app --reload
```

### Desktop client

```powershell
cd desktop
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

### Run tests

```powershell
cd backend
pip install pytest httpx
$env:TESTING = "true"
$env:DATABASE_URL = "sqlite+pysqlite:///:memory:"
pytest tests/ -v
```

---

## API endpoints (v1)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/health` | Health check → `{"status": "ok"}` |
| POST | `/v1/enroll` | Enroll face (multipart `file` + optional `label`) |
| POST | `/v1/search?k=5` | Search face (multipart `file`) |
| GET | `/v1/persons/{id}` | Get person + embeddings |
| DELETE | `/v1/persons/{id}` | Soft‑delete person |
| GET | `/v1/index/stats` | Index statistics |
| POST | `/v1/index/rebuild` | Rebuild index (`index_type` + `params`) |

---

## Embedding backends

| Backend | Env value | Extra deps | Notes |
|---------|-----------|------------|-------|
| Dummy | `dummy` | none | Fixed vector `[1,0,…]` — tests & demo |
| InsightFace | `insightface` | `insightface onnxruntime` | Full pipeline |
| Torch | `torch` | `torch opencv-python-headless` | Custom IR‑ResNet |
| ONNX | `onnx` | `onnxruntime opencv-python-headless` | Skeleton — BYO model |

Set in `.env`:

```env
EMBEDDING_BACKEND=dummy
EMBEDDING_DIM=512
```

---

## Scripts

```powershell
# Compute embeddings from dataset (uses EMBEDDING_BACKEND from .env)
python scripts/compute_embeddings.py --dataset path/to/dataset --batch-size 500

# Rebuild FAISS index from DB
python scripts/build_index.py --index-type hnsw

# Benchmark search
python scripts/benchmark_search.py --dataset path/to/dataset --k 5 --samples 100
```

---

## Project structure

```
├── backend/
│   ├── app/
│   │   ├── api/          # routes + schemas (stable v1 contract)
│   │   ├── core/         # config, logging
│   │   ├── db/           # models, session, migrations
│   │   └── services/
│   │       ├── embeddings/   # interface + dummy/insightface/torch/onnx
│   │       ├── face/         # detector, align, quality
│   │       ├── index/        # VectorIndex ABC + FAISS adapter
│   │       └── storage/      # SQLAlchemy repositories
│   ├── tests/
│   ├── requirements.txt      # core (no ML)
│   └── requirements-ml.txt   # optional ML libs
├── desktop/              # PySide6 GUI client
├── scripts/              # batch processing CLIs
├── tools/run_all.ps1     # one‑click Docker startup
├── .env.example          # default config (dummy backend)
├── docker-compose.yml
└── pyproject.toml        # black/ruff/pytest config
```

---

## Architecture: Ports & Adapters

* **EmbeddingExtractor** (port) — `extract_embedding(bytes) → ndarray`
* **VectorIndex** (port) — `add / search / save / load / stats / train`
* **Repositories** (port) — `PersonRepo`, `EmbeddingRepo`, `IndexSnapshotRepo`

Adapters are selected at startup via config. Backend code never imports ML
libraries directly — only through the extractor factory.

## Notes

* Original face images are **never stored**. Only embeddings + metadata.
* Strict single‑face policy by default (0 or >1 faces → HTTP 422).
* Index is in‑memory for speed; persisted to disk + DB snapshot on changes.
