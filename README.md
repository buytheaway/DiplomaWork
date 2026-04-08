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
- `compare` mode for side-by-side pipeline evaluation
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

### 1. Start the backend

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

### 2. Start the desktop client

Open a second terminal:

```powershell
cd desktop
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
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
docker compose up --build
```

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
- `Enroll` registers a person in `pretrained`, `custom`, or `both`
- `Compare` runs both pipelines on the same image and shows side-by-side results

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
- deleting a person rebuilds the affected pipeline index to keep search results consistent
- FAISS index loading is strict: a missing `.map.json` sidecar is treated as a broken index state
- operational routes require `API_KEY`
- delete and rebuild require `ADMIN_API_KEY`

## Main API routes

- `GET /v1/health`
- `POST /v1/enroll`
- `POST /v1/search`
- `POST /v1/search/compare`
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

Training is CUDA-only in the current branch.

## Tests

Backend tests:

```powershell
cd backend
.\.venv\Scripts\activate
python -m pytest tests -q
```

Quick syntax check:

```powershell
python -m compileall backend/app desktop/app training
```

## Troubleshooting

### `No module named uvicorn`

You are probably inside `desktop/.venv`. Activate `backend/.venv` and run the backend there.

### `GET /v1/persons` returns `500`

Check `DATABASE_URL` and run Alembic migrations for the backend environment.

### Compare mode is unavailable

Open `GET /v1/health` and verify that both `pretrained` and `custom` pipelines are available.

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
