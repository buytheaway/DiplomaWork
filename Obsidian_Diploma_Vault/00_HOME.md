# DiplomaWork Vault

This vault is the working knowledge base for the pre-defense and final defense.

## Quick entry points

- [[01_Project/01_Project_Overview]]
- [[01_Project/02_Architecture]]
- [[01_Project/03_Backend]]
- [[01_Project/04_Desktop]]
- [[01_Project/05_Data_and_Models]]
- [[01_Project/06_API_and_Endpoints]]
- [[02_Defense/01_Demo_Script]]
- [[02_Defense/02_Defense_QA]]
- [[02_Defense/03_Risks_and_Limitations]]
- [[03_Research/01_Datasets]]
- [[03_Research/02_Training]]
- [[03_Research/03_Benchmarking]]
- [[04_Diagrams/00_Diagrams_Index]]

## What is here

- `01_Project` - system structure and main components
- `02_Defense` - demo order and defense answers
- `03_Research` - datasets, training and benchmarking
- `04_Diagrams` - Mermaid diagrams for architecture and workflows
- `99_Templates` - note templates

## Short project summary

DiplomaWork is a biometric face search system.

It includes:

- backend on FastAPI;
- desktop client on PySide6;
- FAISS vector search;
- database storage for metadata and encrypted embeddings;
- a research branch for training and evaluation;
- a comparative branch for side-by-side runtime experiments.

## What to emphasize first during defense

1. The project is a system, not only a model.
2. Face search is split into extraction and approximate nearest-neighbor retrieval.
3. Metadata is stored in DB, while FAISS handles vector search.
4. `Enroll` is strict, `Search` and live webcam support multiple faces.
5. The runtime includes auth, encrypted storage and audit logging.
