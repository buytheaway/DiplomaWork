# Architecture

Связанные схемы:

- [[04_Diagrams/01_System_Architecture_Diagram]]
- [[04_Diagrams/02_Search_Flow_Diagram]]
- [[04_Diagrams/03_Enroll_Flow_Diagram]]
- [[04_Diagrams/04_Compare_Mode_Diagram]]
- [[04_Diagrams/05_Live_Webcam_Diagram]]

## Высокоуровневая схема

Система разделена на четыре ключевых уровня:

- desktop client;
- backend API;
- storage layer;
- vector search layer.

## Поток данных

1. Desktop отправляет изображение или кадр камеры в backend.
2. Backend валидирует файл.
3. Extractor находит лицо и вычисляет embedding.
4. Index manager отправляет embedding в FAISS.
5. FAISS возвращает top-k ближайших совпадений.
6. Backend дополняет ответ metadata из БД.
7. Desktop показывает результаты пользователю.

## Почему архитектура удобная

- UI не зависит от конкретной модели.
- API не зависит от конкретной реализации desktop.
- БД не используется как векторный движок.
- Index layer можно пересобирать и сравнивать.
- Можно держать несколько pipeline параллельно.

## Основные папки проекта

- `backend/` — API, runtime, index, storage
- `desktop/` — операторский интерфейс
- `training/` — обучение и оценка
- `deploy/model_bundle/` — внешний bundle
- `scripts/` — служебные скрипты

## Что говорить, если спросят "почему не только БД?"

Потому что обычная SQL-БД хорошо хранит записи, но плохо подходит для быстрого поиска ближайших 512-мерных векторов на больших объёмах. Поэтому metadata и embeddings хранятся в БД, а nearest-neighbor поиск вынесен в FAISS.
