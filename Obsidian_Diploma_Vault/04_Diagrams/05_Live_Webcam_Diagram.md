# Live Webcam Diagram

Related notes:

- [[01_Project/04_Desktop]]
- [[01_Project/03_Backend]]
- [[02_Defense/01_Demo_Script]]

```mermaid
flowchart TD
    A[Desktop opens local camera] --> B[Preview stays local]
    B --> C[Timer picks a frame]
    C --> D{Current mode}
    D -- Search --> E[Send frame to /v1/search]
    D -- Compare --> F[Send frame to /v1/search/compare]
    E --> G[Backend detects one or more faces]
    F --> G
    G --> H[Backend computes per-face embeddings and searches FAISS]
    H --> I[Backend returns detected_faces and top matches]
    I --> J[Desktop groups faces and draws overlays]
    J --> K[Operator sees labels, scores and unknown faces]
```

## Important wording

This is a near real-time webcam workflow, not full video streaming to the server.
The preview stays local in the desktop app.
The backend receives selected frames at intervals and can process multiple faces in one frame.
