# Fast Biometric Face Search

Production-quality diploma prototype for face-based biometric search with FastAPI, PostgreSQL, FAISS, and a PySide6 desktop client.

## Quick start

1) Copy `.env.example` to `.env` and adjust if needed.
2) Start services:

```bash
docker-compose up --build
```

The API is available at `http://localhost:8000`.

## Run on Windows (PowerShell)

```powershell
Set-Location -Path "C:\Users\mukha\OneDrive\Documents\GitHub\DiplomaWork"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up --build
```

Health check (PowerShell):

```powershell
Invoke-WebRequest http://localhost:8000/v1/health
```

Docs:

```powershell
Start-Process http://localhost:8000/docs
```

## Backend (local)

```bash
cd backend
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

## Desktop app

```bash
cd desktop
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

## Compute embeddings from dataset

Dataset format: `dataset/person_id_or_label/*.jpg`

```bash
python scripts/compute_embeddings.py --dataset /path/to/dataset
```

## Rebuild index

```bash
python scripts/build_index.py --index-type hnsw
```

## Benchmark search

```bash
python scripts/benchmark_search.py --dataset /path/to/dataset --k 5 --samples 100
```

## Training (from scratch)

This repo includes a minimal PyTorch training pipeline under `training/` to train a custom
face embedding model with ArcFace loss. Training does **not** store images in the backend.

Default plan for your setup:
- Train on CASIA-WebFace (train) and use LFW for validation only
- `batch_size=64`, `epochs=20`

Run training (PowerShell):
```powershell
python .\training\train.py --config .\training\config.yaml
```

Evaluate weights:
```powershell
python .\training\eval.py --config .\training\config.yaml --weights .\training\outputs\checkpoint_epoch_020.pth
```

Use trained weights in backend (set in `.env`):
```
EMBEDDING_BACKEND=torch
TORCH_MODEL_PATH=training/outputs/checkpoint_epoch_020.pth
TORCH_MODEL_ARCH=ir50
TORCH_DEVICE=cuda
TORCH_USE_FP16=true
```

## API endpoints

- `GET /v1/health`
- `POST /v1/enroll` (multipart: `file`, `label`)
- `POST /v1/search?k=5` (multipart: `file`)
- `GET /v1/persons/{id}`
- `DELETE /v1/persons/{id}`
- `GET /v1/index/stats`
- `POST /v1/index/rebuild`

## Notes

- Original face images are never stored. Only embeddings and minimal metadata are persisted.
- Index files are saved to `backend/data/index` by default.
