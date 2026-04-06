# API and Endpoints

## Health

`GET /v1/health`

Возвращает:

- `status`
- `default_pipeline`
- `available_pipelines`
- `models`
- `backends`
- `strict_single_face_enroll`
- `multi_face_search`

## Enroll

`POST /v1/enroll`

Назначение:

- зарегистрировать одного человека по изображению

Особенности:

- одно лицо обязательно;
- при нескольких лицах запрос отклоняется;
- сохраняется person + embedding.

## Search

`POST /v1/search`

Назначение:

- поиск по одному pipeline

Возвращает:

- top-k matches;
- `score`;
- `distance`;
- `pipeline`;
- `latency_ms`;
- `detected_faces`.

## Compare

`POST /v1/search/compare`

Назначение:

- сравнение `pretrained` и `custom` на одном входе

Полезно для:

- benchmark;
- демонстрации различий pipeline;
- анализа latency и confidence.

## Persons

`GET /v1/persons`

Назначение:

- просмотр зарегистрированных записей.

## Index

`GET /v1/index/stats`

- статистика индекса;
- pipeline-aware operational info.

`POST /v1/index/rebuild`

- пересборка индекса;
- используется в maintenance сценариях.
