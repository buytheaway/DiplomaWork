# Fast Biometric Face Search

Дипломный проект — быстрый биометрический поиск по лицу.

**Стек:** FastAPI · PostgreSQL / SQLite · FAISS · PySide6 desktop client.

**Ключевой принцип:** ML-модель — это *плагин*. Система работает
с `EMBEDDING_BACKEND=dummy` (без ML-зависимостей). Для настоящего
распознавания — переключить на `insightface` или `onnx` через `.env`.

---

## Быстрый старт (Windows, без Docker)

### 1. Backend

```powershell
cd backend
pip install -r requirements.txt

# Создать .env в корне репозитория (по умолчанию — dummy бэкенд):
Copy-Item ..\.env.example ..\.env -Force

# Запустить:
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Проверка: `http://127.0.0.1:8000/v1/health`
Swagger: `http://127.0.0.1:8000/docs`

### 2. Desktop

В отдельном терминале:

```powershell
cd desktop
pip install -r requirements.txt
python -m app.main
```

При запуске в строке состояния отобразится статус подключения к бэкенду.

Если бэкенд недоступен — десктоп покажет предупреждение, но запустится.

### 3. Тесты

```powershell
cd backend
$env:TESTING = "true"
$env:DATABASE_URL = "sqlite+pysqlite:///:memory:"
python -m pytest tests/ -v
```

---

## Docker

```powershell
# Скопировать конфиг
if (-not (Test-Path .env)) { Copy-Item .env.example .env }

# Запуск с dummy бэкендом
docker compose up --build

# С ML-зависимостями (insightface)
docker compose build --build-arg ML_BACKEND=insightface
# Поменять в .env: EMBEDDING_BACKEND=insightface
docker compose up
```

---

## Ручное тестирование (curl / PowerShell)

Все запросы к `http://127.0.0.1:8000`.

```powershell
# Health check
Invoke-WebRequest http://127.0.0.1:8000/v1/health | Select-Object -Expand Content

# Enroll (зарегистрировать лицо)
curl -X POST http://127.0.0.1:8000/v1/enroll -F "file=@photo.jpg" -F "label=Alice"

# Search (поиск по лицу)
curl -X POST "http://127.0.0.1:8000/v1/search?k=5" -F "file=@photo.jpg"

# Persons (получить / удалить)
curl http://127.0.0.1:8000/v1/persons/<person_id>
curl -X DELETE http://127.0.0.1:8000/v1/persons/<person_id>

# Index stats / rebuild
curl http://127.0.0.1:8000/v1/index/stats
curl -X POST http://127.0.0.1:8000/v1/index/rebuild -H "Content-Type: application/json" -d '{"index_type":"hnsw","params":{"m":32,"ef_construction":200,"ef_search":64}}'
```

---

## API endpoints (v1)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/health` | `{"status": "ok", "embedding_backend": "…"}` |
| POST | `/v1/enroll` | Multipart `file` + optional `label` |
| POST | `/v1/search?k=5` | Multipart `file` → результаты + decision |
| GET | `/v1/persons/{id}` | Получить person + embeddings |
| DELETE | `/v1/persons/{id}` | Soft-delete |
| GET | `/v1/index/stats` | Статистика индекса |
| POST | `/v1/index/rebuild` | `{"index_type": "hnsw", "params": {…}}` |

---

## Embedding backends

| Backend | Значение `.env` | Доп. зависимости | Статус |
|---------|-----------------|-----------------|--------|
| Dummy | `dummy` | нет | Всегда работает (тесты, демо) |
| InsightFace | `insightface` | `insightface onnxruntime` | Рабочий, buffalo_l |
| ONNX | `onnx` | `onnxruntime opencv-python-headless` | Рабочий, нужны `.onnx` файлы |
| Torch | `torch` | `torch opencv-python-headless` | **Экспериментальный** — чекпоинты недообучены |

### Настройка InsightFace

```env
EMBEDDING_BACKEND=insightface
EMBEDDING_DIM=512
MODEL_NAME=buffalo_l
```

Модели скачаются автоматически при первом запуске.

### Настройка ONNX

```env
EMBEDDING_BACKEND=onnx
ONNX_DETECTOR_PATH=models/scrfd_10g_bnkps.onnx
ONNX_EMBEDDER_PATH=models/w600k_r50.onnx
```

Файлы `.onnx` нужно скачать и положить по указанным путям.

### Настройка Torch (экспериментальный)

```env
EMBEDDING_BACKEND=torch
TORCH_MODEL_PATH=../training/outputs/checkpoint_epoch_005.pth
TORCH_DEVICE=cpu
DETECTION_BACKEND=opencv
```

> ⚠ Текущие чекпоинты дали 0% val accuracy за 5 эпох.
> Для реального использования нужно переобучить модель на большем датасете.

---

## Match threshold и decision

Ответ `/v1/search` включает автоматическое решение:

```json
{
  "k": 5,
  "model": "insightface_buffalo_l",
  "results": [ … ],
  "best_score": 0.87,
  "threshold_used": 0.4,
  "best_match_above_threshold": true,
  "decision": "match"
}
```

- `decision: "match"` — best_score >= MATCH_THRESHOLD
- `decision: "unknown"` — ниже порога или пустые результаты

Настройка порога:

```env
MATCH_THRESHOLD=0.4
```

Чем выше — тем строже. Если порог слишком высокий, система будет возвращать `unknown` даже для правильных лиц.

---

## Важные ограничения

- **Оригинальные фото не хранятся.** Только эмбеддинги + метаданные.
- **Strict single-face policy:** 0 лиц → 422, больше 1 лица → 422.
- **Индекс в памяти**, сохраняется на диск при изменениях.

---

## Структура проекта

```
├── backend/
│   ├── app/
│   │   ├── api/          # routes + schemas
│   │   ├── core/         # config, logging
│   │   ├── db/           # models, session, migrations
│   │   └── services/
│   │       ├── embeddings/   # dummy / insightface / torch / onnx
│   │       ├── face/         # detector, align, quality
│   │       ├── index/        # FAISS adapter
│   │       └── storage/      # SQLAlchemy repositories
│   ├── tests/
│   ├── requirements.txt
│   └── requirements-ml-*.txt
├── desktop/              # PySide6 GUI
├── training/             # IR-ResNet обучение (экспериментальное)
├── scripts/              # CLI утилиты
├── .env.example          # дефолтный конфиг
└── docker-compose.yml
```
