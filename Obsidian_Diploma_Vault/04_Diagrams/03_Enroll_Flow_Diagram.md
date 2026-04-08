# Enroll Flow Diagram

Related notes:

- [[01_Project/03_Backend]]
- [[01_Project/04_Desktop]]
- [[01_Project/06_API_and_Endpoints]]

```mermaid
flowchart TD
    A[Operator chooses image, label and pipeline] --> B[Desktop sends POST /v1/enroll]
    B --> C[Backend validates API key and upload]
    C --> D[Detector finds faces]
    D --> E{Exactly one face?}
    E -- No faces --> F[Return 422]
    E -- More than one --> G[Return 422]
    E -- Yes --> H[Compute embedding]
    H --> I[Create new person record]
    I --> J[Store encrypted embedding in DB]
    J --> K[Update selected pipeline index]
    K --> L[Save encrypted index snapshot]
    L --> M[Write audit log]
    M --> N[Return enroll response]
```

## Key point

`Enroll` is intentionally stricter than `Search`.
The current backend requires exactly one face and creates a new person record for that enroll request.
This keeps the gallery cleaner and avoids ambiguous person-to-face mapping.
