# Demo Script

## Recommended order

1. Open `Dashboard`
2. Show that backend is online
3. Show available pipelines
4. Open `Face Search`
5. Run a normal `Search`
6. Show `Enroll` with a single-face image
7. Show `Compare`
8. Open `Database`
9. Open `Logs`
10. If stable, show live webcam mode

## What to say during the demo

### Dashboard

"Here the operator sees backend health, available pipelines and whether the system is ready for work."

### Search

"The backend receives an image, extracts embeddings and uses FAISS to search the biometric gallery."

### Enroll

"Enroll is intentionally strict: exactly one face is required so the gallery stays clean."

### Compare

"Here we compare two pipelines on the same input and inspect latency and result quality side by side."

### Database

"Here we see registered records and metadata. Delete actions are admin-only."

### Logs

"This screen is used for operational maintenance: backend state, index state and rebuild actions."

### Live webcam

"The preview stays local. The desktop sends selected frames to the backend and can process multiple faces in one frame."

## If time is short

Show in this order:

1. Dashboard
2. Search
3. Compare
4. Database
