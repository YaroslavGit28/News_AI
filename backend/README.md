# Backend (FastAPI)

## Основные сервисы
- FastAPI REST API (`app/main.py`) + кэш реальных RSS-источников (`services/feed_cache.py`)
- Celery-задачи для агрегации RSS (`app/tasks/ingest.py`)
- Сервисы суммаризации, тематизации и рекомендаций (`app/services/*`)

## Запуск
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Celery worker (опционально):
```bash
celery -A app.tasks.ingest.celery_app worker --loglevel=info
```
