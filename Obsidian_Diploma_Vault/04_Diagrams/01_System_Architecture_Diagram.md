# System Architecture Diagram

Related notes:

- [[01_Project/02_Architecture]]
- [[01_Project/03_Backend]]
- [[01_Project/04_Desktop]]
- [[01_Project/06_API_and_Endpoints]]

```mermaid
flowchart LR
    U[Operator] --> D[PySide6 Desktop]
    D --> API[FastAPI Backend]

    API --> AUTH[API key / Admin key]
    API --> REG[Pipeline Registry]
    REG --> PRE[Pretrained pipeline]
    REG --> CUS[Custom pipeline]

    PRE --> EXT1[Extractor]
    CUS --> EXT2[Extractor]

    EXT1 --> IDX1[FAISS index: pretrained]
    EXT2 --> IDX2[FAISS index: custom]

    API --> DB[(PostgreSQL / SQLite)]
    API --> AUD[Audit log]
    API --> SNAP[Encrypted snapshot files]

    DB --> META[Person metadata + encrypted embeddings]
    IDX1 --> API
    IDX2 --> API
    API --> RESP[Search / Enroll / Compare response]
    RESP --> D
```

## What this diagram shows

- desktop and backend are separated;
- the runtime supports multiple pipelines;
- each pipeline has its own index;
- the database stores metadata and encrypted embeddings;
- FAISS does vector search, not SQL;
- audit logging and encrypted snapshots are part of the runtime, not an external afterthought.
