# Модель данных (MVP)

## Таблица `sources`
- `id` — PK
- `name`
- `url`
- `category`
- `reliability_score`

## Таблица `articles`
- `id` — PK
- `title`
- `url`
- `summary`
- `topics` — список (хранится как JSON или vector array)
- `entities`
- `sentiment`
- `published_at`
- `created_at`
- `source_id` — FK → `sources`

## Дополнительно
- `user_profiles` (интересы, настройки фильтров)
- `feedback_events` (клики, лайки, скрытия)
