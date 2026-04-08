# API and Endpoints

## Security

Operational routes require `X-API-Key`.

- `API_KEY` is used for normal operator actions;
- `ADMIN_API_KEY` is used for admin actions;
- `GET /v1/health` is the public health probe;
- when auth is enabled, API docs are disabled.

## Health

`GET /v1/health`

Returns:

- `status`
- `default_pipeline`
- `available_pipelines`
- `models`
- `backends`
- `strict_single_face_enroll`
- `multi_face_search`

## Enroll

`POST /v1/enroll`

Purpose:

- register one person from one image

Behavior:

- exactly one face is required;
- if multiple faces are present, the request is rejected;
- the backend creates a new person record;
- the embedding is stored encrypted;
- the selected pipeline index is updated.

## Search

`POST /v1/search`

Purpose:

- search in one selected pipeline

Returns:

- top-k matches;
- `score`;
- `distance`;
- `pipeline`;
- `latency_ms`;
- `detected_faces`.

This route supports multiple faces in one image.

## Compare

`POST /v1/search/compare`

Purpose:

- compare `pretrained` and `custom` on the same input

Useful for:

- benchmark;
- pipeline comparison;
- latency and confidence analysis.

## Persons

`GET /v1/persons`

- list registered person records.

`GET /v1/persons/{person_id}`

- show one record and its embeddings.

`DELETE /v1/persons/{person_id}`

- admin-only;
- soft-deletes the person and rebuilds affected indexes.

## Index

`GET /v1/index/stats`

- pipeline-aware index statistics.

`POST /v1/index/rebuild`

- admin-only;
- rebuilds the selected pipeline index.
