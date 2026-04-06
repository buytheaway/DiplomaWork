# Backend

Связанные схемы:

- [[04_Diagrams/01_System_Architecture_Diagram]]
- [[04_Diagrams/02_Search_Flow_Diagram]]
- [[04_Diagrams/03_Enroll_Flow_Diagram]]
- [[04_Diagrams/04_Compare_Mode_Diagram]]

## Технология

Backend построен на FastAPI.

Точка входа:

- `backend/app/main.py`

## Что делает backend

- поднимает runtime pipeline;
- инициализирует индексы;
- обрабатывает enroll и search запросы;
- отдаёт health/status;
- работает с person records;
- отдаёт статистику индекса;
- умеет rebuild индекса.

## Основные маршруты

- `GET /v1/health`
- `POST /v1/enroll`
- `POST /v1/search`
- `POST /v1/search/compare`
- `GET /v1/persons`
- `GET /v1/index/stats`
- `POST /v1/index/rebuild`

## PipelineRegistry

Registry поднимает и держит доступные pipeline:

- `pretrained`
- `custom`

Также он:

- знает default pipeline;
- умеет отдавать available pipelines;
- загружает index snapshots;
- создаёт runtime для каждого pipeline.

## Важные правила обработки лица

- `Enroll` допускает только одно лицо.
- `Search` допускает несколько лиц.
- `0 faces` на enroll -> `422`
- `>1 faces` на enroll -> `422`
- invalid image -> `400`

## Что backend хранит

По умолчанию backend не хранит сырые изображения как основной operational storage.

Хранятся:

- `person`
- `embedding`
- `index snapshot`
