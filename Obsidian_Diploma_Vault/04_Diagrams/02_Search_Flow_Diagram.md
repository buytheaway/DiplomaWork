# Search Flow Diagram

Связано с:

- [[01_Project/02_Architecture]]
- [[01_Project/03_Backend]]
- [[01_Project/04_Desktop]]
- [[01_Project/06_API_and_Endpoints]]

```mermaid
flowchart TD
    A[User selects image] --> B[Desktop sends request to /v1/search]
    B --> C[Backend validates image]
    C --> D[Extractor detects one or more faces]
    D --> E[Embedding is computed for each face]
    E --> F[FAISS searches top-k nearest vectors]
    F --> G[Backend loads metadata from DB]
    G --> H[Builds SearchResponse]
    H --> I[Desktop shows summary]
    I --> J[Desktop shows top matches and detected faces]
```

## Ключевая мысль

Search — это workflow из нескольких этапов: validation, detection, embedding extraction, ANN retrieval, metadata hydration и UI presentation.
