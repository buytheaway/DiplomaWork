# Fast Biometric Face Search

Desktop biometric face search system with a FastAPI backend, FAISS vector search,
PostgreSQL/SQLite storage, and a PySide6 operator client.

The project is built around two runtime pipelines:

- `custom`: final Torch face-embedding pipeline, model name
  `torch_insightface_iresnet100`.
- `pretrained`: ONNX/InsightFace baseline, model name `onnx_w600k_r50`.

The custom pipeline is the main project runtime. The pretrained pipeline is kept
as a comparison baseline and operational reference.

## Current Final State

Final custom model configuration:

| Field | Value |
|---|---|
| Runtime model | `torch_insightface_iresnet100` |
| Architecture | `insightface_iresnet100` |
| Checkpoint path | `custom_torch_candidate_bundle/model.pth` |
| Preprocessing | `runtime_fallback_center_crop` |
| Color order | `RGB` |
| Normalization | `[-1, 1]` |
| TTA | `hflip` |
| Selected threshold | `0.205047` |

`model.pth` is intentionally not tracked in git. It must be present locally for
the custom Torch runtime to start.

Final LFW biometric verification result:

| Metric | Value |
|---|---:|
| Valid pairs | 6000 / 6000 |
| Skipped pairs | 0 |
| Accuracy | 0.990500 |
| EER | 0.015000 |
| EER threshold | 0.154398 |
| Selected threshold | 0.205047 |
| FAR at selected threshold | 0.001667 |
| FRR at selected threshold | 0.017333 |
| TAR@FAR=0.01 | 0.984667 |
| TAR@FAR=0.001 | 0.981000 |

Important interpretation: LFW is a real biometric verification protocol. It is
not a desktop/live webcam top-1 identification score and it is not a FAISS
scalability benchmark.

## Project Layout

```text
backend/                       FastAPI backend
desktop/                       PySide6 desktop client
models/                        local ONNX model assets, ignored when binary
custom_torch_candidate_bundle/ runtime custom model manifest and docs
training/                      training and evaluation utilities
scripts/                       import, benchmark, evaluation, index scripts
tools/                         local launch/helper scripts
docs/                          tracked technical documentation
reports/                       tracked final summaries, generated raw outputs ignored
data/                          local enroll/import data, ignored
datasets/                      local datasets, ignored
external_models/               downloaded candidate models, ignored
```

Do not commit datasets, face images, checkpoints, FAISS indexes, `.env` files,
SQLite databases, PostgreSQL dumps, embedding files, or cache directories.

## Main Runtime Features

- FastAPI backend with health, search, enroll, persons, database stats, and
  index routes.
- PySide6 desktop client with Dashboard, Face Search, Database, and Logs tabs.
- Separate FAISS index per pipeline.
- Paginated Database tab with total identity/template/index counts.
- Multi-face search toggle.
- Live webcam search path with fast-only search settings.
- API-key protection for operator and admin actions.
- Encrypted embedding payloads and encrypted FAISS snapshot files.
- Offline FAISS index builder for large scale databases.

## Full Local Demo Launcher

For the final scale-demo setup, use:

```powershell
.\start_scale_full.bat
```

The launcher starts:

- Docker PostgreSQL database service.
- Local FastAPI backend with scale DB/index environment.
- Local PySide6 desktop client.

The launcher opens separate backend and desktop windows. Secrets are read from
local environment/configuration and must not be committed.

Latest recorded scale smoke state:

| Metric | Value |
|---|---:|
| Active identities | 5,741 |
| Active templates | 2,015,076 |
| Custom templates | 2,015,037 |
| Pretrained templates | 39 |
| Custom indexed vectors | 2,015,037 |
| Pretrained indexed vectors | 39 |

The scale database contains real-image-derived embeddings plus manually enrolled
profiles. Synthetic vector benchmarks are separate and must not be mixed with
biometric accuracy claims.

## Manual Backend Run

Use a backend virtual environment:

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

Minimum runtime environment variables:

```dotenv
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@127.0.0.1:55432/biometric_vggface2_2m_real
DEFAULT_PIPELINE=custom
ENABLE_PRETRAINED_PIPELINE=true
ENABLE_CUSTOM_PIPELINE=true

PRETRAINED_BACKEND=onnx
CUSTOM_BACKEND=torch
EMBEDDING_BACKEND=onnx

ONNX_DETECTOR_PATH=models/det_10g.onnx
ONNX_EMBEDDER_PATH=models/w600k_r50.onnx

TORCH_MODEL_PATH=custom_torch_candidate_bundle/model.pth
TORCH_MODEL_ARCH=insightface_iresnet100
TORCH_MODEL_NAME=torch_insightface_iresnet100
TORCH_PREPROCESS=runtime_fallback_center_crop
TORCH_TTA=hflip
TORCH_DEVICE=cuda
TORCH_USE_FP16=true

CUSTOM_MATCH_THRESHOLD=0.205047
CUSTOM_LIVE_MATCH_THRESHOLD=0.205047

PRETRAINED_INDEX_PATH=tmp/scale_index_pretrained/pretrained.faiss
CUSTOM_INDEX_PATH=tmp/scale_index_custom/custom.faiss
INDEX_PATH=tmp/scale_index_custom/custom.faiss
AUTO_SAVE_INDEX=false

API_KEY=REPLACE_WITH_GENERATED_OPERATOR_KEY
ADMIN_API_KEY=REPLACE_WITH_GENERATED_ADMIN_KEY
DATA_ENCRYPTION_KEY=REPLACE_WITH_GENERATED_32_BYTE_BASE64_KEY
SNAPSHOT_ENCRYPTION_KEY=REPLACE_WITH_GENERATED_32_BYTE_BASE64_KEY
RATE_LIMIT_ENABLED=false
```

If CUDA is not available:

```dotenv
TORCH_DEVICE=cpu
TORCH_USE_FP16=false
```

Generate local secrets:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
python -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
```

Never commit `.env`, `.env.local`, `.env.docker`, real keys, database files, or
model checkpoints.

## Manual Desktop Run

Use a separate desktop virtual environment:

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

Do not start the backend from `desktop/.venv`.

## Docker

Docker is used for PostgreSQL in the final local setup. The backend itself runs
locally by default.

```powershell
docker compose --env-file .env.docker up -d db
```

`.env.docker` is local-only and must not contain public credentials in git.

## API Keys

The backend expects two keys outside `TESTING`:

- `API_KEY` for normal operator actions.
- `ADMIN_API_KEY` for admin actions such as delete and index rebuild.

The desktop sends `X-API-Key`. Delete and rebuild actions use the admin key.

## Main API Routes

- `GET /v1/health`
- `POST /v1/enroll`
- `POST /v1/search`
- `GET /v1/persons`
- `DELETE /v1/persons/{person_id}`
- `GET /v1/database/stats`
- `GET /v1/index/stats`
- `POST /v1/index/rebuild`

## Face Processing Rules

- Enroll requires exactly one face.
- Search can process one face or multiple faces depending on the request flag.
- Live webcam defaults to fast single/largest-face behavior.
- Invalid image returns a validation error.
- No face detected returns a no-face response.
- The backend stores encrypted embeddings, not raw photos by default.

## Embedding Space Rule

Embedding spaces must not be mixed.

Old `torch_ir50` embeddings cannot be converted into
`torch_insightface_iresnet100` embeddings. Existing persons/metadata can be
reused, but new model embeddings must be created from original images or by
re-enrollment, then indexed in a separate custom FAISS index.

## Evaluation Tracks

Keep these tracks separate:

| Track | Data | Measures | Does not measure |
|---|---|---|---|
| LFW biometric verification | Real labeled face pairs | FAR, FRR, EER, TAR@FAR, threshold behavior | FAISS scalability |
| Real-image embedding benchmark | Embeddings extracted from real images | Extraction success, FAISS build/search latency | FAR/FRR/EER |
| Synthetic 1M/2M benchmark | Synthetic normalized vectors | FAISS scalability, build time, index size, overlap | Biometric accuracy |
| Desktop/live smoke | Runtime backend workflow | End-to-end demo behavior and latency | Formal LFW verification |

## LFW Evaluation

Run custom LFW verification:

```powershell
python scripts\evaluate_lfw_verification.py `
  --lfw-root handoff_lfw_eval\lfw `
  --pairs-file handoff_lfw_eval\lfw\pairs.txt `
  --pipeline custom `
  --output-dir reports\biometric_eval\lfw_custom
```

Run pretrained baseline:

```powershell
python scripts\evaluate_lfw_verification.py `
  --lfw-root handoff_lfw_eval\lfw `
  --pairs-file handoff_lfw_eval\lfw\pairs.txt `
  --pipeline pretrained `
  --output-dir reports\biometric_eval\lfw_pretrained
```

Generate comparison:

```powershell
python scripts\compare_lfw_metrics.py `
  --custom reports\biometric_eval\lfw_custom\metrics.json `
  --pretrained reports\biometric_eval\lfw_pretrained\metrics.json `
  --output reports\biometric_eval\lfw_comparison.md
```

Tracked summaries:

- `docs/benchmarks/lfw_biometric_verification_results.md`
- `reports/final_evaluation_summary.md`
- `reports/final_tables_for_diploma.md`

## Real-Image Embedding Benchmark

The real-image benchmark extracts embeddings from actual image files and builds
an isolated FAISS index.

Latest recorded result:

| Metric | Value |
|---|---:|
| Real images scanned | 206,802 |
| Images attempted | 10,000 |
| Embeddings created | 9,802 |
| Skipped multiple-face images | 198 |
| Embedding dim | 512 |
| Index type | HNSW |
| Index size | 21.76 MB |
| Search p50 | 0.141150 ms |
| Search p95 | 0.194645 ms |
| Search p99 | 0.338031 ms |

Run example:

```powershell
python scripts\benchmark_real_image_embeddings.py `
  --sources datasets\celeba_faces handoff_lfw_eval\lfw data\new_custom_enroll `
  --output-dir reports\real_image_embedding_benchmark `
  --max-images 10000 `
  --batch-size 512 `
  --n-queries 100 `
  --top-k 10 `
  --index-type hnsw
```

## Synthetic FAISS Scalability Benchmark

The synthetic benchmark uses generated L2-normalized 512D vectors. It is a
retrieval scalability stress test only.

Tracked 1M/2M results are documented under:

```text
docs/benchmarks/scale_2m/
```

Interpretation:

- HNSW has stronger retrieval agreement but larger memory/build cost.
- IVF-PQ is compact and fast.
- IVF-PQ at small scales with poor `top_k_overlap` must be described as
  unsuitable or poorly tuned for small datasets, not as biometric accuracy.

## Importing New Custom Faces

Folder mode:

```text
data/new_custom_enroll/
  Person Label/
    img1.jpg
    img2.jpg
```

CSV mode:

```csv
person_id,label,image_path
1,Daniyar,path\to\img1.jpg
1,Daniyar,path\to\img2.jpg
```

Use:

```powershell
python scripts\import_new_custom_faces.py `
  --database-url $env:DATABASE_URL `
  --folder data\new_custom_enroll `
  --pipeline custom `
  --model-name torch_insightface_iresnet100 `
  --rebuild-index
```

This creates new embeddings for the new custom model. It does not delete or
convert old `torch_ir50` embeddings.

## Offline Scale Index Builder

For large PostgreSQL databases, do not use backend `/v1/index/rebuild` for very
large indexes. Use the offline builder:

```powershell
python scripts\build_scale_index_from_db.py `
  --database-url $env:DATABASE_URL `
  --pipeline custom `
  --model-name torch_insightface_iresnet100 `
  --index-type hnsw `
  --index-path $env:CUSTOM_INDEX_PATH `
  --batch-size 50000 `
  --yes
```

The builder writes backend-compatible encrypted snapshot files and an
`index_snapshots` database row.

## Tests

Run checks from the repository root:

```powershell
python -m ruff check backend desktop scripts training tests
python -m compileall -q backend\app desktop\app scripts training tests
python -m pytest tests\test_biometric_metrics.py -q
```

Backend tests should run in test mode so local `.env` does not force the real
custom runtime:

```powershell
$env:TESTING='true'
$env:DEFAULT_PIPELINE='pretrained'
$env:ENABLE_PRETRAINED_PIPELINE='true'
$env:ENABLE_CUSTOM_PIPELINE='true'
$env:EMBEDDING_BACKEND='dummy'
$env:PRETRAINED_BACKEND='dummy'
$env:CUSTOM_BACKEND='dummy'
$env:DETECTION_BACKEND='none'
$env:CUSTOM_DETECTION_BACKEND='none'
$env:MATCH_THRESHOLD='0.4'
$env:PRETRAINED_MATCH_THRESHOLD=''
$env:CUSTOM_MATCH_THRESHOLD=''
$env:CUSTOM_LIVE_MATCH_THRESHOLD=''
python -m pytest backend\tests -q
```

## Security and Privacy Notes

- Do not commit `.env`, `.env.local`, `.env.docker`, database files, model
  checkpoints, datasets, raw images, FAISS indexes, or embeddings.
- API-key comparison uses timing-safe comparison.
- Admin actions require `ADMIN_API_KEY`.
- Embeddings and index snapshots are encrypted at rest by the application.
- Encryption does not replace filesystem, backup, and deployment access
  control.
- The in-memory rate limiter is suitable for local MVP/demo only. Distributed
  deployments need Redis, an API gateway, or another external limiter.
- Soft delete is implemented. Hard purge is not a public endpoint.

## Troubleshooting

### `No module named uvicorn`

You are probably inside `desktop/.venv`. Activate `backend/.venv`.

### Desktop says request rejected

Check that desktop `API_KEY` and `ADMIN_API_KEY` match backend values.

### Database tab cannot delete a person

Delete requires the admin key.

### `/docs` does not open

Expected when API-key auth is enabled. Authenticated mode disables docs.

### A person exists but is not recognized

Check:

- the same pipeline/model was used for enroll and search;
- the correct FAISS index is loaded for that model;
- the image is clear, single-face, and close enough;
- the best score passes the configured threshold;
- old `torch_ir50` vectors are not being used for the new
  `torch_insightface_iresnet100` runtime.
