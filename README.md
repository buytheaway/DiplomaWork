# Fast Biometric Face Search

Production-quality diploma prototype for face-based biometric search with FastAPI, PostgreSQL, FAISS, and a PySide6 desktop client.

## Quick start

1) Copy `.env.example` to `.env` and adjust if needed.
2) Start services:

```bash
docker-compose up --build
```

The API is available at `http://localhost:8000`.

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
