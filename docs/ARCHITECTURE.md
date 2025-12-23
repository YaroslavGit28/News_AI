# Архитектура Persona News

## Слои
1. **Ingestion** — Celery-задачи (`app/tasks/ingest.py`) и синхронный `FeedCache` (`services/feed_cache.py`) опрашивают RSS-ленты (BBC, Meduza, The Verge, CNN, NYTimes, Reuters, Guardian, DW, РБК, Lenta, TechCrunch, Wired и др.), кэшируют результат на 10 минут.
2. **Processing** — сервисы нормализации, суммаризации (`services/summarizer.py`), тематизации (`services/topic_classifier.py`), извлечение сущностей.
3. **API** — FastAPI (`app/main.py`) отдаёт здоровье, источники, персонализированную ленту, принимает обратную связь (лайк/дизлайк/скрыть/undo).
4. **UI** — Next.js-приложение (`frontend/`) с дэшбордом и карточками тем.
5. **Хранилища** — сейчас in-memory, в проде PostgreSQL + Redis + объектное для архивов.

## Поток данных
Источник → Очередь → Обогащение → База → API → Клиент → Обратная связь.

## Точки расширения
- замена простых сервисов на LLM/встраивания (OpenAI, Azure, локальные модели);
- добавление аналитики и профилей пользователей;
- расширение ingestion новыми коннекторами (Telegram, Twitter, Slack).
