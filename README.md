# Fast Biometric Face Search

Desktop biometric face search system with a FastAPI backend, FAISS vector search, and a PySide6 operator client.

## What the project includes

- `backend/` FastAPI application with search, enroll, persons, health, and index routes
- `desktop/` PySide6 operator client
- `models/` runtime model assets
- `training/` research and training utilities
- `scripts/` import, benchmark, and export helpers

Main runtime features:

- `pretrained` pipeline
- `custom` pipeline
- strict single-face enroll
- multi-face search
- near real-time webcam search in the desktop client
- separate FAISS index per pipeline

## Project layout

```text
backend/                  FastAPI backend
desktop/                  PySide6 desktop client
models/                   runtime ONNX assets
training/                 training and evaluation utilities
scripts/                  import, benchmark, export scripts
tools/                    local helper scripts
deploy/model_bundle/      local external bundle artifacts
Obsidian_Diploma_Vault/   project notes and diagrams
```

## Local run

Use two separate virtual environments:

- `backend/.venv` for FastAPI and backend dependencies
- `desktop/.venv` for the PySide6 desktop client

Do not start the backend from `desktop/.venv`.

Before the first local run, create your local backend config:

```powershell
Copy-Item .env.local.example .env
```

Then replace every `REPLACE_WITH_*` value in `.env`. The example values are placeholders and must not be used in production or in demos with real biometric data.

Generate API keys:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Generate encryption keys:

```powershell
python -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
```

### 1. Start the backend

The backend reads configuration from `DiplomaWork\.env` or from environment
variables. A plain `python -m uvicorn app.main:app ...` will fail if these
values are missing.

For the local defense/demo runtime, `.env` must include at least:

```dotenv
DATABASE_URL=sqlite+pysqlite:///./backend_local_test3.db
DEFAULT_PIPELINE=custom
ENABLE_PRETRAINED_PIPELINE=true
ENABLE_CUSTOM_PIPELINE=true

PRETRAINED_BACKEND=onnx
CUSTOM_BACKEND=torch
EMBEDDING_BACKEND=onnx

ONNX_DETECTOR_PATH=models/det_10g.onnx
ONNX_EMBEDDER_PATH=models/w600k_r50.onnx

TORCH_MODEL_PATH=diplomcheckbackup/training/outputs_medium_lfw_finetune/best_lfw.pth
TORCH_MODEL_ARCH=ir50
TORCH_DEVICE=cuda
TORCH_USE_FP16=true
MATCH_THRESHOLD=0.348

CUSTOM_DETECTION_BACKEND=yolo
YOLO_MODEL_PATH=diplomcheckbackup/training/detector_runs/face_yolo_ft1/weights/best.pt
CUSTOM_ALLOW_CENTER_CROP=true
MIN_DET_SCORE=0.5

PRETRAINED_INDEX_PATH=backend/data/index/pretrained.faiss
CUSTOM_INDEX_PATH=backend/data/index/custom.faiss
INDEX_PATH=backend/data/index/custom.faiss
AUTO_SAVE_INDEX=true

API_KEY=REPLACE_WITH_GENERATED_OPERATOR_KEY
ADMIN_API_KEY=REPLACE_WITH_GENERATED_ADMIN_KEY
DATA_ENCRYPTION_KEY=REPLACE_WITH_GENERATED_32_BYTE_BASE64_KEY
SNAPSHOT_ENCRYPTION_KEY=REPLACE_WITH_GENERATED_32_BYTE_BASE64_KEY
RATE_LIMIT_ENABLED=false
```

If CUDA is not available, use:

```dotenv
TORCH_DEVICE=cpu
TORCH_USE_FP16=false
```

Do not commit `.env`. Model/checkpoint files, SQLite databases, FAISS indexes,
embeddings, and biometric images are runtime artifacts and must stay outside git.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
curl.exe http://127.0.0.1:8000/v1/health
```

The current local demo workspace may also have generated helper scripts in
`$env:TEMP`. They are not part of the repository, but they are convenient when
the local model bundle and keys are already configured:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\start_diplomawork_backend_custom_best_lfw.ps1"
```

### 2. Start the desktop client

Open a second terminal:

```powershell
cd desktop
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
$env:API_BASE_URL="http://127.0.0.1:8000"
$env:API_KEY="your-operator-key"
$env:ADMIN_API_KEY="your-admin-key"
$env:API_TIMEOUT_SEC="45"
$env:LIVE_MAX_WIDTH="640"
$env:LIVE_JPEG_QUALITY="75"
python -m app.main
```

If the local helper script exists, it can start the desktop with the same demo
keys:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\start_diplomawork_desktop_custom_demo.ps1"
```

## API keys

The backend now expects two keys outside `TESTING`:

- `API_KEY` for normal operator actions
- `ADMIN_API_KEY` for admin actions such as delete and index rebuild

For the desktop client set the same values in the shell before launch:

```powershell
$env:API_KEY="your-operator-key"
$env:ADMIN_API_KEY="your-admin-key"
python -m app.main
```

The backend uses `X-API-Key` for both operator and admin requests.
The desktop automatically sends `ADMIN_API_KEY` for delete and rebuild actions.

## Docker

`tools/run_all.ps1` starts Docker Compose for the backend and database:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run_all.ps1
```

This script starts:

- `db`
- `backend`

It does not start the desktop client. Launch the desktop locally from `desktop/.venv`.

Manual Docker run:

```powershell
if (-not (Test-Path .env.docker)) { Copy-Item .env.docker.example .env.docker }
docker compose --env-file .env.docker up --build
```

Docker Compose is configured for local development only. Replace the `.env.docker` placeholders, including the Postgres password and backend keys, before running with any real biometric data.

Useful URLs:

- `http://127.0.0.1:8000/v1/health`
- `http://127.0.0.1:8000/v1/index/stats`

`/docs` is available only when auth is disabled. With `API_KEY` enabled, docs are intentionally turned off.

## Desktop workflow

Main pages:

- `Dashboard`
- `Face Search`
- `Database`
- `Logs`

### Face Search modes

- `Search` searches one pipeline at a time
- `Enroll` registers a person in both available pipelines so the demo database stays synchronized

### Live webcam

The webcam mode is near real-time:

- preview stays local in the desktop app
- one frame is sent at the selected interval
- the backend returns detections and matches
- the desktop draws face overlays and updates the summary

This is not continuous 30 FPS recognition. It is operator-oriented periodic scanning.

## Face processing rules

- `Enroll` requires exactly one face
- `Search` can process multiple faces
- invalid image -> `400`
- no face detected -> `422`
- multiple faces during enroll -> `422`

The backend does not store raw photos by default. It stores:

- `person`
- encrypted `embedding`
- encrypted `index snapshot`
- `audit log`

## Runtime notes

- each pipeline has its own FAISS index
- old encrypted index snapshots are pruned by `INDEX_SNAPSHOT_RETENTION` to reduce disk growth
- deleting a person rebuilds the affected pipeline index to keep search results consistent
- FAISS index loading is strict: a missing `.map.json` sidecar is treated as a broken index state
- operational routes require `API_KEY`
- delete and rebuild require `ADMIN_API_KEY`
- in-memory rate limiting protects search, enroll, rebuild, and delete routes; configure it with `RATE_LIMIT_*`
- distributed deployments need Redis or another external rate limiter because in-memory buckets are per process
- index snapshots contain biometric template index data, even when encrypted; protect backups and filesystem access carefully

## Main API routes

- `GET /v1/health`
- `POST /v1/enroll`
- `POST /v1/search`
- `GET /v1/persons`
- `DELETE /v1/persons/{person_id}`
- `GET /v1/index/stats`
- `POST /v1/index/rebuild`

## Training and research utilities

The `training/` directory contains research utilities for the custom branch:

- training scripts
- LFW evaluation
- ONNX export helpers
- dataset preparation scripts

The custom Torch IR-50 branch is the proposed project pipeline for the defense
runtime. The pretrained ONNX/InsightFace path is kept as a comparison baseline
and operational reference. Custom model quality claims require valid training
logs and labeled verification results.

Use prepared CelebA identity folders for custom IR-50 fine-tuning:

```text
datasets/celeba_faces/train/<identity>/*.jpg
datasets/celeba_faces/val/<identity>/*.jpg
```

Do not use LFW for training. LFW is reserved for final FAR/FRR/EER verification
evaluation. DigiFace1M may be used for synthetic warmstart or scale data, but it
does not replace real-face fine-tuning on CelebA.

## Evaluation and benchmarks

Synthetic FAISS retrieval benchmark:

```powershell
python scripts\benchmark_retrieval.py `
  --sizes 100 1000 10000 `
  --dim 512 `
  --n-queries 100 `
  --latency-warmup 10 `
  --latency-runs 100 `
  --hnsw-ef-search 64 `
  --ivfpq-nlist 16 `
  --ivfpq-m 16 `
  --ivfpq-nbits 4 `
  --ivfpq-nprobe 8 `
  --output training\outputs\retrieval_benchmark.json
```

The tracked PR 2 benchmark artifacts are stored in `docs/benchmarks/`. The
synthetic retrieval metric is `top_k_overlap@K`, not biometric identification
hit@K and not biometric accuracy.

LFW biometric verification evaluator:

```powershell
python scripts\evaluate_lfw_verification.py `
  --lfw-root handoff_lfw_eval\lfw `
  --pairs-file handoff_lfw_eval\lfw\pairs.txt `
  --pipeline custom `
  --output-dir reports\biometric_eval\lfw_custom
```

Tracked summary results are documented in
`docs/benchmarks/lfw_biometric_verification_results.md`. The custom Torch IR-50
pipeline is the proposed project pipeline; the pretrained ONNX/InsightFace
pipeline is the external comparison baseline. Do not claim the custom model is
state-of-the-art or better than the pretrained baseline unless the reported
metrics prove it.

Keep the two evaluation tracks separate:

- biometric quality: LFW labeled-pair metrics such as FAR, FRR, EER, and
  TAR@FAR;
- retrieval scalability: FAISS behavior on 1M/2M embeddings, reported with
  retrieval latency, build time, index size, and `top_k_overlap@K`.

If the custom checkpoint, preprocessing, or embedding model changes, rebuild all
custom stored embeddings and custom FAISS indexes. Old vectors live in the old
embedding space and are not compatible with the new checkpoint.

## Tests

Run checks from the repository root:

```powershell
python -m pytest backend\tests -q
python -m pytest training -q
python -m ruff check backend training scripts
python -m compileall -q backend\app desktop\app training scripts
```

## Security and privacy notes

Before using real biometric data:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
python -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
```

- Replace all `REPLACE_WITH_*` placeholders in `.env`, `.env.local`, or `.env.docker`.
- Do not use example configs with real biometric data.
- API-key comparison uses timing-safe comparison.
- In-memory rate limiting is configured with `RATE_LIMIT_ENABLED`, `RATE_LIMIT_SEARCH_PER_MIN`, `RATE_LIMIT_ENROLL_PER_MIN`, and `RATE_LIMIT_ADMIN_PER_MIN`.
- The in-memory limiter is suitable for the local MVP only; distributed deployments need Redis, an API gateway, or another external limiter.
- FAISS snapshot retention is configured with `INDEX_SNAPSHOT_RETENTION`; values `<= 0` disable pruning.
- Index snapshots and embeddings are sensitive biometric template artifacts and still require controlled filesystem access, backups, and deployment-level protection.
- Soft delete is implemented. A hard purge endpoint is not implemented yet.

## Troubleshooting

### `No module named uvicorn`

You are probably inside `desktop/.venv`. Activate `backend/.venv` and run the backend there.

### `GET /v1/persons` returns `500`

Check `DATABASE_URL` and run Alembic migrations for the backend environment.

### The desktop cannot connect to the backend

Check:

- backend is running
- `API_BASE_URL` points to the correct host
- `API_KEY` matches the backend, if enabled
- `ADMIN_API_KEY` is set if you use delete or rebuild

### `/docs` does not open

This is expected when `API_KEY` is enabled. The backend disables docs in authenticated mode.

### A person exists in the database but is not recognized

Check:

- the same pipeline was used for enroll and search
- the index was rebuilt if records were changed manually
- the search image is clear enough
- the best score actually passes the configured threshold
