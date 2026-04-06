# System Architecture Diagram

Связано с:

- [[01_Project/02_Architecture]]
- [[01_Project/03_Backend]]
- [[01_Project/04_Desktop]]
- [[01_Project/06_API_and_Endpoints]]

```mermaid
flowchart LR
    U[Operator] --> D[PySide6 Desktop]
    D --> API[FastAPI Backend]

    API --> REG[Pipeline Registry]
    REG --> PRE[Pretrained pipeline]
    REG --> CUS[Custom pipeline]

    PRE --> EXT1[Extractor]
    CUS --> EXT2[Extractor]

    EXT1 --> IDX1[FAISS index: pretrained]
    EXT2 --> IDX2[FAISS index: custom]

    API --> DB[(PostgreSQL / SQLite)]
    IDX1 --> API
    IDX2 --> API
    DB --> API

    API --> RESP[Search / Enroll / Compare response]
    RESP --> D
```

## Что показывает схема

- desktop и backend разделены;
- runtime поддерживает несколько pipeline;
- для каждого pipeline есть свой индекс;
- БД хранит metadata и embeddings;
- поиск идёт через FAISS, а не через SQL как vector engine.
