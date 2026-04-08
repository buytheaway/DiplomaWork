# Architecture

Related diagrams:

- [[04_Diagrams/01_System_Architecture_Diagram]]
- [[04_Diagrams/02_Search_Flow_Diagram]]
- [[04_Diagrams/03_Enroll_Flow_Diagram]]
- [[04_Diagrams/04_Compare_Mode_Diagram]]
- [[04_Diagrams/05_Live_Webcam_Diagram]]

## High-level structure

The system is split into five major layers:

- desktop client;
- backend API;
- pipeline runtime;
- storage layer;
- vector search layer.

## Data flow

1. Desktop sends an image or a selected webcam frame to the backend.
2. Backend validates the request and API key.
3. The extractor detects faces and computes embeddings.
4. The index manager searches FAISS.
5. The backend loads person metadata from the database.
6. The backend builds a response and writes an audit event.
7. Desktop shows per-face results to the operator.

## Why this architecture is useful

- UI is not tied to one model implementation.
- API is not tied to one desktop implementation.
- SQL storage is not misused as the vector engine.
- The index layer can be rebuilt and benchmarked independently.
- Multiple pipelines can run side by side.
- Security concerns are explicit: auth, encrypted storage and audit are separate concerns, not hidden inside model code.

## Main project directories

- `backend/` - API, runtime, index and storage logic
- `desktop/` - operator interface
- `training/` - research and evaluation
- `deploy/model_bundle/` - local external artifact, not part of the trusted repo payload
- `scripts/` - helper scripts

## Why not use only SQL?

Regular SQL storage is good for records and metadata, but poor for fast nearest-neighbor search in 512-dimensional space.
That is why metadata and encrypted embeddings live in the database, while nearest-neighbor search is delegated to FAISS.
