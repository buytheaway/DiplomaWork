# Backend

Related diagrams:

- [[04_Diagrams/01_System_Architecture_Diagram]]
- [[04_Diagrams/02_Search_Flow_Diagram]]
- [[04_Diagrams/03_Enroll_Flow_Diagram]]
- [[04_Diagrams/04_Compare_Mode_Diagram]]

## Technology

The backend is implemented with FastAPI.

Entry point:

- `backend/app/main.py`

## What the backend does

- initializes runtime pipelines;
- initializes and loads FAISS indexes;
- processes enroll and search requests;
- exposes health and index status;
- manages person records;
- rebuilds indexes;
- enforces API-key based access;
- writes audit logs.

## Main routes

- `GET /v1/health`
- `POST /v1/enroll`
- `POST /v1/search`
- `POST /v1/search/compare`
- `GET /v1/persons`
- `DELETE /v1/persons/{person_id}`
- `GET /v1/index/stats`
- `POST /v1/index/rebuild`

## PipelineRegistry

The registry manages the currently available pipelines:

- `pretrained`
- `custom`

It also:

- knows the default pipeline;
- reports available pipelines;
- loads the latest snapshots;
- creates runtime objects per pipeline.

## Face-processing rules

- `Enroll` accepts exactly one face.
- `Search` accepts multiple faces.
- `0 faces` on enroll -> `422`
- `>1 faces` on enroll -> `422`
- invalid image -> `400`

## Storage and security

The backend does not use raw images as its default operational storage.

Stored entities:

- `person`
- `embedding`
- `index snapshot`
- `audit log`

Security behavior:

- `API_KEY` protects normal routes;
- `ADMIN_API_KEY` protects admin actions such as delete and rebuild;
- embeddings are encrypted before writing to DB;
- FAISS snapshot files are encrypted on disk;
- audit entries are retained with a configurable retention period.
