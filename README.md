# Fast Biometric Face Search
xx`
Дипломный проект: быстрый биометрический поиск по лицу.

## Что есть в проекте

- `pretrained` pipeline: baseline на `buffalo_l` через ONNX
- `custom` pipeline: дополнительная ONNX модель (из `model_bundle`)
- FastAPI backend
- FAISS индекс отдельно для каждого pipeline
- PostgreSQL через Docker Compose или SQLite локально
- PySide6 desktop клиент
- multi-face search для обеих моделей
- strict single-face enroll по умолчанию

## Быстрый локальный запуск

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Проверка:

```powershell
curl.exe http://127.0.0.1:8000/v1/health
```

### Desktop

```powershell
cd desktop
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

## Docker Compose

Без custom-модели контейнер стартует в baseline-режиме. Это безопасный дефолт.

```powershell
if (-not (Test-Path .env.docker)) { Copy-Item .env.docker.example .env.docker }
docker compose up --build
```

Или коротко:

```powershell
.\tools\run_all.ps1
```

Что делает Docker-профиль:

- поднимает `db`
- поднимает `backend`
- монтирует `models/` в контейнер
- монтирует `training/outputs/` для кастомных весов

Если хочешь включить custom pipeline в Docker:

1. положи ONNX-модели в `model_bundle/models/`
2. открой `.env.docker`
3. выставь:

```env
ENABLE_CUSTOM_PIPELINE=true
CUSTOM_BACKEND=onnx
ONNX_DETECTOR_PATH=/app/model_bundle/models/det_10g.onnx
ONNX_EMBEDDER_PATH=/app/model_bundle/models/w600k_r50.onnx
```

Проверка:

```powershell
curl.exe http://127.0.0.1:8000/v1/health
```

## Desktop режимы

- `Pretrained`
- `Custom`
- `Compare both`

В режиме поиска backend умеет возвращать результаты сразу по нескольким лицам. В desktop это видно по колонке `Face`.

## Правила face processing

- `Enroll`: строго одно лицо
- `Search`: можно несколько лиц
- `0` лиц -> `422`
- `>1` лицо на `Enroll` -> `422`
- невалидное изображение -> `400`

Фото сервер не хранит. Сохраняются только:

- `person`
- `embedding`
- `index snapshot`

## CelebA для custom модели

Подготовленный формат:

```text
datasets/celeba_faces/
  train/<identity>/*.jpg
  val/<identity>/*.jpg
  test/<identity>/*.jpg
```

### Автоскачивание через torchvision

Нужен `gdown`:

```powershell
python -m pip install gdown
```

Скачать CelebA:

```powershell
@'
from torchvision.datasets import CelebA
CelebA(
    root=r'c:\Users\mukha\OneDrive\Documents\GitHub\DiplomaWork\datasets\celeba_raw',
    split='train',
    download=True,
)
'@ | python -
```

Разложить в формат проекта:

```powershell
python training\tools\prepare_celeba.py --source-root datasets\celeba_raw\celeba --output-root datasets\celeba_faces --clean
```

## Обучение custom модели

`training/config.yaml` уже настроен под CelebA:

- `batch_size = 64`
- `epochs = 20`
- `train_dir = datasets/celeba_faces/train`
- `val_dir = datasets/celeba_faces/val`

Запуск:

```powershell
python training\train.py --config training\config.yaml --device cuda
```

Быстрый smoke run:

```powershell
python training\train.py --config training\config.yaml --device cuda --epochs 1
```

Оценка:

```powershell
python training\eval.py --config training\config.yaml --weights training\outputs\checkpoint_epoch_020.pth --device cuda --num-workers 0
```

## Важное про pretrained baseline

`pretrained` pipeline сейчас используется как замороженный baseline для сравнения.

То есть:

- `buffalo_l` используется для инференса и benchmark
- локальное обучение в этом репозитории идёт для `custom` PyTorch модели

Если нужен fine-tune baseline-модели, это уже отдельный train pipeline, а не текущий ONNX runtime.

## Тесты

```powershell
cd backend
python -m pytest tests -q
```

## Полезные endpoints

- `GET /v1/health`
- `POST /v1/enroll`
- `POST /v1/search`
- `POST /v1/search/compare`
- `GET /v1/persons`
- `GET /v1/index/stats`
- `POST /v1/index/rebuild`
