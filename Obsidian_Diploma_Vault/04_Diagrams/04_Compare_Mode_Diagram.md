# Compare Mode Diagram

Related notes:

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
        P1-->>API: per-face matches + latency
    and Custom
        API->>P2: detect + embed + search
        P2-->>API: per-face matches + latency
    end
    API-->>D: comparisons[] + detected faces
    D-->>U: side-by-side results by score and latency
```

## Why this mode exists

- compare latency;
- compare pipeline behavior on the same input;
- show baseline vs comparative branch;
- use the same UI mode as a benchmark and demo tool.
