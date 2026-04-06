# Training

## Зачем в проекте training-ветка

Training-ветка нужна для исследовательской части:

- обучение собственной модели;
- evaluation;
- подготовка датасетов;
- export артефактов;
- benchmark.

## Что в неё входит

- `training/train.py`
- `training/eval_lfw.py`
- `training/config.yaml`
- `training/losses/`
- `training/models/`
- `scripts/export_onnx.py`
- `scripts/benchmark_retrieval.py`

## Что важно проговаривать

Training subsystem — это отдельный исследовательский слой, а не единственный operational runtime системы.

## Что можно сказать коротко

“В репозитории есть отдельная исследовательская подсистема, которая позволяет обучать и оценивать собственные модели, а также экспортировать их для дальнейшего использования.”

## Что не стоит путать

- training results;
- runtime baseline;
- external model bundle.

Это три разные вещи.
