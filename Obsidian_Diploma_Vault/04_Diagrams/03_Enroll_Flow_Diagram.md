# Enroll Flow Diagram

Связано с:

- [[01_Project/03_Backend]]
- [[01_Project/04_Desktop]]
- [[01_Project/06_API_and_Endpoints]]

```mermaid
flowchart TD
    A[User chooses image and label] --> B[Desktop sends request to /v1/enroll]
    B --> C[Backend validates file]
    C --> D[Detect faces]
    D --> E{Exactly one face?}
    E -- No faces --> F[Return 422]
    E -- More than one --> G[Return 422]
    E -- Yes --> H[Compute embedding]
    H --> I[Create or reuse person record]
    I --> J[Store embedding]
    J --> K[Update pipeline index]
    K --> L[Return enroll response]
```

## Ключевая мысль

Enroll специально более строгий, чем search. Это сделано для чистоты базы и корректного соответствия one person -> one valid facial sample.
