# Live Webcam Diagram

Связано с:

- [[01_Project/04_Desktop]]
- [[01_Project/03_Backend]]
- [[02_Defense/01_Demo_Script]]

```mermaid
flowchart TD
    A[Desktop opens local camera] --> B[Preview frame is shown locally]
    B --> C[Timer selects frame]
    C --> D[Frame sent to backend]
    D --> E[Face detection and embedding extraction]
    E --> F[FAISS search]
    F --> G[Backend returns detected faces and matches]
    G --> H[Desktop draws overlays]
    H --> I[Operator sees labels, scores and status]
```

## Важная формулировка

Это near real-time webcam workflow, а не тяжёлый постоянный видеострим на сервер. Preview живёт локально, а backend получает выбранные кадры по таймеру.
