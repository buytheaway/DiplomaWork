# Compare Mode Diagram

Связано с:

- [[01_Project/03_Backend]]
- [[01_Project/04_Desktop]]
- [[01_Project/06_API_and_Endpoints]]
- [[03_Research/03_Benchmarking]]

```mermaid
sequenceDiagram
    participant U as Operator
    participant D as Desktop
    participant API as Backend
    participant P1 as Pretrained pipeline
    participant P2 as Custom pipeline

    U->>D: Choose image and Compare mode
    D->>API: POST /v1/search/compare
    par Pretrained
        API->>P1: detect + embed + search
        P1-->>API: matches + latency
    and Custom
        API->>P2: detect + embed + search
        P2-->>API: matches + latency
    end
    API-->>D: comparisons[]
    D-->>U: compare results, score and latency
```

## Зачем нужен этот режим

- сравнить latency;
- сравнить поведение pipeline;
- показать baseline vs comparative branch;
- использовать режим как benchmark and demo tool.
