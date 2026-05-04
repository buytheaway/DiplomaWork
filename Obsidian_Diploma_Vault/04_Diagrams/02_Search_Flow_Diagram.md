# Search Flow Diagram

Related notes:

- [[01_Project/02_Architecture]]
- [[01_Project/03_Backend]]
- [[01_Project/04_Desktop]]
- [[01_Project/06_API_and_Endpoints]]

```mermaid
flowchart TD
    A[Operator selects image or live frame] --> B[Desktop sends POST /v1/search]
    B --> C[Backend validates API key and upload]
    C --> D[Extractor detects one or more faces]
    D --> E[Compute one embedding per detected face]
    E --> F[FAISS searches top-k nearest vectors for each face]
    F --> G[Backend loads person metadata from DB]
    G --> H[Apply threshold and build per-face response]
    H --> I[Write audit log]
    I --> J[Desktop shows summary, detected faces and top matches]
```

## Key point

`Search` is a multi-step workflow: validation, multi-face detection, embedding extraction, approximate nearest-neighbor retrieval,
metadata hydration, decision logic, audit logging and UI presentation.
