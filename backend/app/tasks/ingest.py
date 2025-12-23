from __future__ import annotations
from datetime import datetime
from celery import Celery
from ..services import summarizer, topic_classifier
from ..datasources.rss import RssSource, fetch, DEFAULT_SOURCES

celery_app = Celery(
    "persona_news",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

SOURCES = DEFAULT_SOURCES[:6]


@celery_app.task
def ingest_sources() -> list[dict]:
    """Fetch + enrich новости. В реальном проекте сохраняли бы в БД."""

    news_items: list[dict] = []
    for source in SOURCES:
        for entry in fetch(source):
            summary = summarizer.Summarizer().summarize(entry["summary"])
            topics = topic_classifier.classify(entry["summary"])
            news_items.append(
                {
                    "title": entry["title"],
                    "url": entry["link"],
                    "summary": summary,
                    "topics": topics,
                    "source": source.name,
                    "published_at": entry.get("published", datetime.utcnow().isoformat()),
                }
            )
    return news_items
