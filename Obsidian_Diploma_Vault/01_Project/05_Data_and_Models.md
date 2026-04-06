# Data and Models

## Что используется в проекте

В проекте есть несколько модельных и датасетных направлений.

## Runtime pipeline

Логически в системе есть:

- `pretrained`
- `custom`
- `compare`

### Pretrained

Baseline используется как стабильный эталон для сравнения.

### Custom

Custom branch используется как comparative / research направление.

### Compare

Позволяет на одном и том же изображении сравнить поведение двух pipeline.

## Training subsystem

Отдельная training-ветка нужна для:

- подготовки датасетов;
- обучения собственной модели;
- evaluation;
- export и benchmark.

## deploy/model_bundle

`deploy/model_bundle` — это внешний bundle артефактов.

Там есть:

- `train.csv`
- `faiss.index`
- `meta.json`
- `best.pt`

Его удобно описывать как external comparative bundle, а не как единственный runtime системы.

## Что говорить аккуратно

Безопасная формулировка:

- baseline pipeline стабилен и используется как operational reference;
- custom branch существует для comparative и research задач;
- архитектура поддерживает несколько источников embeddings и несколько runtime path.
