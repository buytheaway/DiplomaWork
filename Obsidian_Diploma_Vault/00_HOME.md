# DiplomaWork Vault

Это стартовая заметка для подготовки к предзащите и защите проекта.

## Быстрый вход

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

## Что здесь лежит

- `01_Project` — устройство системы и основные компоненты
- `02_Defense` — сценарий показа и ответы на вопросы
- `03_Research` — датасеты, обучение, метрики
- `04_Diagrams` — Mermaid-схемы для архитектуры и workflow
- `99_Templates` — заготовки для новых заметок

## Коротко о проекте

DiplomaWork — это система быстрого поиска лиц в биометрической базе данных.

Система состоит из:

- backend на FastAPI;
- desktop-клиента на PySide6;
- FAISS для векторного поиска;
- БД для хранения metadata и embeddings;
- training-ветки для исследовательской части;
- comparative pipeline для сравнения baseline и custom branch.

## Что проговаривать на защите в первую очередь

1. Проект — это не только модель, а полноценная система.
2. Поиск по лицу разделён на extraction и ANN retrieval.
3. БД хранит metadata и embeddings, а FAISS делает быстрый поиск.
4. `Enroll` строгий, `Search` поддерживает multiple faces.
5. Есть desktop, live webcam, compare mode и index maintenance.
