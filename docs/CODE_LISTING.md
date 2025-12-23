# Листинг исходного кода Persona News

Полный текст ключевых файлов проекта для документации.

---

## `backend/app/config.py`

```python
from functools import lru_cache
from pydantic import BaseModel, Field
import os


class Settings(BaseModel):
    app_name: str = Field(default="Persona News API")
    environment: str = Field(default=os.getenv("ENV", "development"))
    database_url: str = Field(default=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./news.db"))
    redis_url: str = Field(default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    openai_api_key: str | None = Field(default=os.getenv("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

---

## `backend/app/datasources/rss.py`

```python
from __future__ import annotations
from typing import Iterable, List
from dataclasses import dataclass
import feedparser


@dataclass
class RssSource:
    name: str
    url: str
    category: str | None = None
    reliability: float = 0.8


def fetch(source: RssSource, limit: int = 20) -> Iterable[dict]:
    feed = feedparser.parse(source.url)
    for entry in feed.entries[:limit]:
        yield {
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "summary": entry.get("summary", ""),
            "published": entry.get("published"),
            "published_parsed": entry.get("published_parsed"),
            "source": source,
        }


DEFAULT_SOURCES: List[RssSource] = [
    RssSource(name="BBC World", url="https://feeds.bbci.co.uk/news/world/rss.xml", category="world", reliability=0.95),
    RssSource(name="CNN World", url="http://rss.cnn.com/rss/edition.rss", category="world", reliability=0.9),
    RssSource(name="NYTimes World", url="https://rss.nytimes.com/services/xml/rss/nyt/World.xml", category="world", reliability=0.92),
    RssSource(name="Reuters World", url="https://feeds.reuters.com/Reuters/worldNews", category="world", reliability=0.93),
    RssSource(name="The Guardian", url="https://www.theguardian.com/world/rss", category="world", reliability=0.9),
    RssSource(name="Euronews", url="https://www.euronews.com/rss?level=theme&name=news", category="world", reliability=0.85),
    RssSource(name="Deutsche Welle", url="https://rss.dw.com/russian", category="society", reliability=0.88),
    RssSource(name="Meduza", url="https://meduza.io/rss/all", category="society", reliability=0.85),
    RssSource(name="Lenta.ru", url="https://lenta.ru/rss/news", category="society", reliability=0.8),
    RssSource(name="RBC", url="https://rssexport.rbc.ru/rbcnews/news/20/full.rss", category="business", reliability=0.85),
    RssSource(name="Bloomberg", url="https://www.bloomberg.com/feed/podcast/etf-report.xml", category="business", reliability=0.9),
    RssSource(name="CNBC", url="https://www.cnbc.com/id/100003114/device/rss/rss.html", category="business", reliability=0.88),
    RssSource(name="Business Insider", url="https://www.businessinsider.com/rss", category="business", reliability=0.84),
    RssSource(name="AP News", url="https://apnews.com/hub/ap-top-news?format=atom", category="world", reliability=0.9),
    RssSource(name="TechCrunch", url="https://techcrunch.com/feed/", category="technology", reliability=0.9),
    RssSource(name="Wired", url="https://www.wired.com/feed/rss", category="technology", reliability=0.9),
    RssSource(name="The Verge", url="https://www.theverge.com/rss/index.xml", category="technology", reliability=0.9),
    RssSource(name="Ars Technica", url="http://feeds.arstechnica.com/arstechnica/index/", category="technology", reliability=0.88),
    RssSource(name="Engadget", url="https://www.engadget.com/rss.xml", category="technology", reliability=0.84),
    RssSource(name="Wired Security", url="https://www.wired.com/category/security/feed", category="technology", reliability=0.87),
    RssSource(name="N+1", url="https://nplus1.ru/rss", category="science", reliability=0.8),
    RssSource(name="MIT Technology Review", url="https://www.technologyreview.com/feed/", category="technology", reliability=0.86),
]
```

---

## `backend/app/main.py`

```python
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .datasources.rss import DEFAULT_SOURCES
from .schemas import HiddenArticleInfo, RecommendationResponse, Source, HealthResponse
from .services.feed_cache import FeedCache, feed_cache
from .services.recommender import SimpleRecommender
from .tasks.ingest import ingest_sources

settings = get_settings()
app = FastAPI(title="Persona News API")
recommender = SimpleRecommender()
cache: FeedCache = feed_cache
USER_FEEDBACK = defaultdict(lambda: {"likes": set(), "dislikes": set(), "hidden": {}})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SOURCES = [
    Source(
        id=index + 1,
        name=src.name,
        url=src.url,
        category=src.category,
        reliability_score=src.reliability,
    )
    for index, src in enumerate(DEFAULT_SOURCES)
]
SOURCE_MAP = {source.id: source for source in SOURCES}


def _get_article(article_id: int) -> Article | None:
    for article in cache.get_articles():
        if article.id == article_id:
            return article
    return None


def _cleanup_hidden(user_id: str) -> dict[int, datetime]:
    prefs = USER_FEEDBACK[user_id]
    hidden: dict[int, datetime] = prefs["hidden"]
    now = datetime.utcnow()
    valid = {aid: ts for aid, ts in hidden.items() if now - ts < timedelta(hours=1)}
    USER_FEEDBACK[user_id]["hidden"] = valid
    return valid

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=datetime.utcnow())


@app.get("/")
def root() -> dict:
    """Показывает полезные ссылки для ручной проверки."""
    return {
        "status": "ok",
        "message": "Persona News API работает. Используйте /feed, /sources, /docs, /health.",
        "docs": "/docs",
        "health": "/health",
        "feed": "/feed",
    }


@app.get("/sources", response_model=list[Source])
def list_sources() -> list[Source]:
    return SOURCES


@app.get("/feed", response_model=RecommendationResponse)
def get_feed(
    user_id: str = Query(default="demo"),
    limit: int = Query(default=25, ge=1, le=100),
) -> RecommendationResponse:
    articles = cache.get_articles()
    hidden = _cleanup_hidden(user_id)
    visible_articles = [article for article in articles if article.id not in hidden]
    personalized = recommender.recommend(user_id=user_id, articles=visible_articles)
    hidden_info = [
        HiddenArticleInfo(
            article_id=aid,
            hidden_at=ts,
            expires_at=ts + timedelta(hours=1),
        )
        for aid, ts in hidden.items()
    ]
    return RecommendationResponse(
        user_id=user_id,
        generated_at=datetime.utcnow(),
        articles=personalized[:limit],
        hidden=hidden_info,
    )


@app.post("/ingest")
def trigger_ingestion() -> dict:
    result = ingest_sources.delay()
    return {"task_id": result.id}


@app.post("/feedback")
def submit_feedback(
    article_id: int = Query(...),
    user_id: str = Query(default="demo"),
    action: str = Query(..., description="like, dislike, hide, undo_hide"),
) -> dict:
    """Обратная связь от пользователя для улучшения рекомендаций"""
    if action not in ["like", "dislike", "hide", "undo_hide"]:
        raise HTTPException(status_code=400, detail="Invalid action. Use: like, dislike, hide, undo_hide")

    prefs = USER_FEEDBACK[user_id]
    article = _get_article(article_id)

    if action in {"like", "dislike"} and not article:
        raise HTTPException(status_code=404, detail="Article not found for feedback")

    message = "ok"
    if action == "like":
        recommender.add_feedback(user_id, article, value=1)
        prefs["likes"].add(article_id)
        prefs["dislikes"].discard(article_id)
        message = "liked"
    elif action == "dislike":
        recommender.add_feedback(user_id, article, value=-1)
        prefs["dislikes"].add(article_id)
        prefs["likes"].discard(article_id)
        message = "disliked"
    elif action == "hide":
        prefs["hidden"][article_id] = datetime.utcnow()
        message = "hidden"
    elif action == "undo_hide":
        removed = prefs["hidden"].pop(article_id, None)
        message = "restored" if removed else "not_hidden"

    return {"status": "success", "message": message, "user_id": user_id}
```

---

## `backend/app/models.py`

```python
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, DateTime, Float, ForeignKey


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    category: Mapped[str | None]
    reliability_score: Mapped[float | None] = mapped_column(Float)

    articles: Mapped[list["Article"]] = relationship("Article", back_populates="source")


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    summary: Mapped[str | None] = mapped_column(String)
    topics: Mapped[str | None] = mapped_column(String)
    entities: Mapped[str | None] = mapped_column(String)
    sentiment: Mapped[float | None] = mapped_column(Float)
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    source: Mapped[Source] = relationship("Source", back_populates="articles")
```

---

## `backend/app/schemas.py`

```python
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class SourceBase(BaseModel):
    name: str
    url: str
    category: str | None = None
    reliability_score: float | None = None


class Source(SourceBase):
    id: int

    class Config:
        from_attributes = True


class ArticleBase(BaseModel):
    title: str
    url: str
    source_id: int
    source_name: Optional[str] = None
    summary: Optional[str] = None
    published_at: Optional[datetime] = None
    topics: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    sentiment: Optional[float] = None
    image_url: Optional[str] = None


class Article(ArticleBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class RecommendationResponse(BaseModel):
    user_id: str
    generated_at: datetime
    articles: List[Article]
    hidden: List["HiddenArticleInfo"] = Field(default_factory=list)


class HiddenArticleInfo(BaseModel):
    article_id: int
    hidden_at: datetime
    expires_at: datetime


RecommendationResponse.model_rebuild()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
```

---

## `backend/app/services/feed_cache.py`

```python
from __future__ import annotations

import hashlib
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, List
from urllib.parse import quote, urlparse

from bs4 import BeautifulSoup

from ..datasources.rss import DEFAULT_SOURCES, RssSource, fetch
from ..schemas import Article
from .summarizer import Summarizer
from .topic_classifier import classify
from .translator import translate_to_ru

summarizer = Summarizer(max_sentences=2)
IMAGE_FALLBACKS = [
    "https://source.unsplash.com/featured/800x400?news",
    "https://source.unsplash.com/featured/800x400?technology",
    "https://source.unsplash.com/featured/800x400?world",
    "https://source.unsplash.com/featured/800x400?business",
    "https://source.unsplash.com/featured/800x400?ai",
]


def _parse_datetime(entry: dict) -> datetime:
    published_struct = entry.get("published_parsed")
    if published_struct:
        return datetime.utcfromtimestamp(time.mktime(published_struct))
    published_text = entry.get("published")
    if published_text:
        try:
            return datetime.fromisoformat(published_text.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass
    return datetime.utcnow()


def _proxy_image(url: str | None) -> str:
    if not url:
        return random.choice(IMAGE_FALLBACKS)
    url = url.replace("http://", "https://")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return random.choice(IMAGE_FALLBACKS)
    path = parsed.path or ""
    if parsed.query:
        path += f"?{parsed.query}"
    proxied = f"https://images.weserv.nl/?url={quote(parsed.netloc + path, safe='/:?=&')}"
    return proxied


def _extract_image(entry: dict) -> str:
    media = entry.get("media_content") or entry.get("media_thumbnail")
    if isinstance(media, list) and media:
        url = media[0].get("url")
        if url:
            return _proxy_image(url)
    enclosure = entry.get("links")
    if isinstance(enclosure, list):
        for item in enclosure:
            if item.get("type", "").startswith("image") and item.get("href"):
                return _proxy_image(item["href"])
    summary = entry.get("summary")
    if summary:
        soup = BeautifulSoup(summary, "html.parser")
        img_tag = soup.find("img")
        if img_tag and img_tag.get("src"):
            return _proxy_image(img_tag["src"])
    return random.choice(IMAGE_FALLBACKS)


def _extract_entities(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-ZА-ЯЁ][a-zа-яёA-ZА-ЯЁ0-9+-]{2,}\b", text)
    seen = []
    for cand in candidates:
        if cand not in seen:
            seen.append(cand)
    return seen[:5]


def _hash_id(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


@dataclass
class FeedCache:
    sources: List[RssSource] = field(default_factory=lambda: DEFAULT_SOURCES)
    ttl_minutes: int = 10
    limit_per_source: int = 25
    _articles: List[Article] = field(default_factory=list)
    _last_updated: datetime | None = None

    def is_expired(self) -> bool:
        if not self._last_updated:
            return True
        return datetime.utcnow() - self._last_updated > timedelta(minutes=self.ttl_minutes)

    def get_articles(self) -> List[Article]:
        if self.is_expired():
            self._articles = self._fetch()
            self._last_updated = datetime.utcnow()
        return self._articles

    def _fetch_source(self, source_id: int, source: RssSource) -> List[Article]:
        batch: List[Article] = []
        for entry in fetch(source, limit=self.limit_per_source):
            url = entry.get("link")
            if not url:
                continue
            title = entry.get("title") or "Без названия"
            raw_summary = entry.get("summary") or title
            summary = summarizer.summarize(raw_summary)
            published_at = _parse_datetime(entry)
            topics = classify(f"{title} {raw_summary}")
            ru_title = translate_to_ru(title)
            ru_summary = translate_to_ru(summary)
            article = Article(
                id=_hash_id(url),
                title=ru_title,
                url=url,
                source_id=source_id,
                source_name=source.name,
                summary=ru_summary,
                topics=topics,
                entities=_extract_entities(ru_title + " " + ru_summary),
                sentiment=None,
                published_at=published_at,
                created_at=datetime.utcnow(),
                    image_url=_extract_image(entry),
            )
            batch.append(article)
        return batch

    def _fetch(self) -> List[Article]:
        collected: dict[str, Article] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(self.sources))) as executor:
            futures = [
                executor.submit(self._fetch_source, source_id, source)
                for source_id, source in enumerate(self.sources, start=1)
            ]
            for future in as_completed(futures):
                for article in future.result():
                    if article.url not in collected:
                        collected[article.url] = article
        ordered = sorted(
            collected.values(),
            key=lambda art: art.published_at or datetime.utcnow(),
            reverse=True,
        )
        return ordered[:300]


feed_cache = FeedCache()
```

---

## `backend/app/services/topic_classifier.py`

```python
from __future__ import annotations
from __future__ import annotations

from typing import Sequence

TOPIC_KEYWORDS: dict[str, set[str]] = {
    "Технологии": {"technology", "tech", "ai", "искусственный", "стартап", "робот", "цифров", "software", "hardware"},
    "Кибербезопасность": {"cyber", "security", "хакер", "кибер", "vulnerability", "breach", "уязвим", "malware", "шифр"},
    "Экономика": {"эконом", "economy", "market", "финанс", "bank", "инфляц", "бирж", "investment", "ipo"},
    "Геополитика": {"полит", "санкции", "конфликт", "war", "geopolitics", "переговор", "election"},
    "Общество": {"общество", "society", "education", "образован", "health", "здоров"},
    "Культура": {"culture", "культура", "art", "музей", "music", "фестиваль", "film", "cinema"},
    "Наука": {"science", "наука", "research", "исслед", "space", "космос", "эксперимент"},
    "Спорт": {"sport", "match", "team", "спорт", "игр", "лига"},
    "Медиа": {"media", "streaming", "youtube", "netflix", "spotify", "hulu", "disney", "tiktok"},
}


def classify(text: str, top_k: int = 3) -> list[str]:
    normalized = text.lower()
    scores: list[tuple[int, str]] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in normalized)
        if score:
            scores.append((score, topic))
    scores.sort(reverse=True, key=lambda item: item[0])
    top_topics = [topic for _, topic in scores[:top_k]]
    return top_topics or ["Общее"]


def batch_classify(texts: Sequence[str], top_k: int = 3) -> list[list[str]]:
    return [classify(text, top_k=top_k) for text in texts]
```

---

## `backend/app/services/recommender.py`

```python
from __future__ import annotations
from datetime import datetime
from collections import defaultdict
from typing import Iterable

from ..schemas import Article


class SimpleRecommender:
    """Смешивает свежесть, интересы пользователя и надежность источника."""

    def __init__(self) -> None:
        self.user_topics: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add_feedback(self, user_id: str, article: Article, value: int = 1) -> None:
        for topic in article.topics:
            self.user_topics[user_id][topic] += value

    def score(self, user_id: str, article: Article) -> float:
        freshness = 1.0
        if article.published_at:
            elapsed = (datetime.utcnow() - article.published_at).total_seconds()
            freshness = max(0.05, 1 - elapsed / (3600 * 24 * 2))
        preference_raw = sum(self.user_topics[user_id].get(t, 0) for t in article.topics)
        preference = 1 + max(-3, preference_raw)
        reliability = 1 + (article.sentiment or 0)
        return freshness * 0.5 + preference * 0.4 + reliability * 0.1

    def recommend(self, user_id: str, articles: Iterable[Article], limit: int = 10) -> list[Article]:
        scored = sorted(articles, key=lambda art: self.score(user_id, art), reverse=True)
        return list(scored[:limit])
```

---

## `backend/app/services/summarizer.py`

```python
from __future__ import annotations
from typing import Sequence


class Summarizer:
    """Простейший summarizer; место для интеграции с ИИ."""

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def summarize(self, text: str) -> str:
        if not text:
            return ""
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        short = ". ".join(sentences[: self.max_sentences])
        if short and not short.endswith("."):
            short += "."
        return short


def batch_summarize(texts: Sequence[str]) -> list[str]:
    summarizer = Summarizer()
    return [summarizer.summarize(text) for text in texts]
```

---

## `backend/app/services/translator.py`

```python
from __future__ import annotations

from functools import lru_cache

from deep_translator import GoogleTranslator


@lru_cache(maxsize=2048)
def translate_to_ru(text: str) -> str:
    if not text:
        return text
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    try:
        translator = GoogleTranslator(source="auto", target="ru")
        return translator.translate(cleaned)
    except Exception:
        return text
```

---

## `backend/app/tasks/ingest.py`

```python
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
```

---

## `backend/requirements.txt`

```
fastapi==0.115.2
uvicorn[standard]==0.30.1
pydantic==2.12.4
SQLAlchemy==2.0.36
asyncpg==0.29.0
redis==5.1.0
celery==5.4.0
httpx==0.27.2
feedparser==6.0.11
beautifulsoup4==4.12.3
python-dotenv==1.0.1
numpy==2.1.2
deep-translator==1.11.4
```

---

## `frontend/app/page.tsx`

```tsx
"use client";

import { useEffect, useState, useMemo } from "react";
import { fetchFeed, Article, HiddenRecord, submitFeedback } from "../lib/api";
import { TopicCard } from "../components/TopicCard";
import { TopicFilter } from "../components/TopicFilter";
import { AppHeader } from "../components/AppHeader";
import { HiddenPanel } from "../components/HiddenPanel";
import { FeedToolbar } from "../components/FeedToolbar";

export default function Page() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTopics, setSelectedTopics] = useState<Set<string>>(new Set());
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [hiddenRecords, setHiddenRecords] = useState<Record<number, HiddenRecord>>({});
  const [timerTick, setTimerTick] = useState(Date.now());
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<"home" | "fresh" | "source">("home");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [savedIds, setSavedIds] = useState<Set<number>>(new Set());
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  useEffect(() => {
    loadFeed();
    // Обновляем каждые 2 минуты
    const interval = setInterval(loadFeed, 120000);
    return () => clearInterval(interval);
  }, []);


  useEffect(() => {
    const interval = setInterval(() => {
      setTimerTick(Date.now());
      setHiddenRecords((prev) => {
        const now = Date.now();
        let mutated = false;
        const next: Record<number, HiddenRecord> = {};
        Object.entries(prev).forEach(([key, record]) => {
          if (new Date(record.expires_at).getTime() > now) {
            next[Number(key)] = record;
          } else {
            mutated = true;
          }
        });
        return mutated ? next : prev;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const stored = window.localStorage.getItem("persona-news:saved");
      if (stored) {
        setSavedIds(new Set(JSON.parse(stored)));
      }
    } catch {
      setSavedIds(new Set());
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem("persona-news:saved", JSON.stringify(Array.from(savedIds)));
  }, [savedIds]);

  const loadFeed = async () => {
    try {
      setLoading((prev) => (articles.length ? prev : true));
      setRefreshing(true);
      const data = await fetchFeed(100);
      setArticles(data.articles);
      const hiddenMap: Record<number, HiddenRecord> = {};
      data.hidden.forEach((item) => {
        hiddenMap[item.article_id] = item;
      });
      setHiddenRecords(hiddenMap);
      setLastUpdated(new Date());
      setErrorMessage(null);
    } catch (error) {
      console.error("Ошибка загрузки ленты:", error);
      const message = error instanceof Error 
        ? error.message 
        : "Не удалось загрузить ленту";
      setErrorMessage(message);
      
      // Если это ошибка подключения, не показываем пустую ленту
      if (message.includes("подключиться") || message.includes("недоступен")) {
        setArticles([]);
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const baseVisibleArticles = useMemo(() => {
    const now = Date.now();
    const query = searchQuery.trim().toLowerCase();
    return articles.filter((article) => {
      const hiddenEntry = hiddenRecords[article.id];
      if (hiddenEntry && new Date(hiddenEntry.expires_at).getTime() > now) {
        return false;
      }
      if (!query) return true;
      const haystack = [
        article.title,
        article.summary,
        article.source_name,
        article.entities.join(" "),
        article.topics.join(" ")
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });
  }, [articles, hiddenRecords, searchQuery]);

  const topicBuckets = useMemo(() => {
    const buckets: Record<string, Article[]> = {};
    baseVisibleArticles.forEach((article) => {
      article.topics.forEach((topic) => {
        if (!buckets[topic]) buckets[topic] = [];
        buckets[topic].push(article);
      });
    });
    Object.values(buckets).forEach((bucket) =>
      bucket.sort((a, b) => {
        const dateA = new Date(a.published_at ?? a.created_at ?? "").getTime();
        const dateB = new Date(b.published_at ?? b.created_at ?? "").getTime();
        return dateB - dateA;
      })
    );
    return buckets;
  }, [baseVisibleArticles]);

  const topicStats = useMemo(() => {
    return Object.entries(topicBuckets)
      .map(([name, list]) => ({ name, count: list.length }))
      .sort((a, b) => b.count - a.count);
  }, [topicBuckets]);

  const filteredArticles = useMemo(() => {
    const now = Date.now();

    let filtered = baseVisibleArticles.filter((article) => {
      if (!selectedTopics.size) return true;
      return article.topics.some((topic) => selectedTopics.has(topic));
    });

    if (selectedTopics.size === 1) {
      const topic = Array.from(selectedTopics)[0];
      const addition = topicBuckets[topic] ?? [];
      for (const article of addition) {
        if (filtered.length >= 5) break;
        const hiddenEntry = hiddenRecords[article.id];
        const hiddenActive = hiddenEntry && new Date(hiddenEntry.expires_at).getTime() > now;
        if (!hiddenActive && !filtered.some((a) => a.id === article.id)) {
          filtered.push(article);
        }
      }
      if (filtered.length < 5) {
        const altTopics = topicStats
          .filter((stat) => stat.name !== topic)
          .slice(0, 3)
          .map((stat) => stat.name);
        altTopics.forEach((alt) => {
          const altList = topicBuckets[alt] ?? [];
          for (const article of altList) {
            if (filtered.length >= 5) break;
            if (!filtered.some((a) => a.id === article.id)) {
              filtered.push(article);
            }
          }
        });
      }
    }

    return filtered;
  }, [baseVisibleArticles, selectedTopics, topicBuckets, topicStats, hiddenRecords]);

  const sortedArticles = useMemo(() => {
    const clone = [...filteredArticles];

    switch (sortBy) {
      case "fresh":
        return clone.sort((a, b) => {
          const dateA = new Date(a.published_at ?? a.created_at ?? "").getTime();
          const dateB = new Date(b.published_at ?? b.created_at ?? "").getTime();
          return dateB - dateA;
        });
      case "source":
        return clone.sort((a, b) => (a.source_name ?? "").localeCompare(b.source_name ?? ""));
      case "home":
      default:
        return clone.sort((a, b) => a.title.localeCompare(b.title, "ru"));
    }
  }, [filteredArticles, sortBy]);

  const handleToggleTopic = (topic: string) => {
    setSelectedTopics((prev) => {
      const next = new Set(prev);
      if (next.has(topic)) {
        next.delete(topic);
      } else {
        next.add(topic);
      }
      return next;
    });
  };

  const handleClearFilter = () => {
    setSelectedTopics(new Set());
  };

  const handleToggleSave = (id: number) => {
    setSavedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleHideArticle = ({ id }: { id: number; title: string }) => {
    setHiddenRecords((prev) => ({
      ...prev,
      [id]: {
        article_id: id,
        hidden_at: new Date().toISOString(),
        expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString()
      }
    }));
  };

  const handleRestoreArticle = async (id: number) => {
    try {
      await submitFeedback(id, "undo_hide");
      setHiddenRecords((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    } catch (error) {
      console.error("Не удалось восстановить статью", error);
    }
  };

  const resolveTitle = (id: number) => articles.find((a) => a.id === id)?.title ?? "Новость";

  const uniqueSortedArticles = useMemo(() => {
    const seen = new Set<number>();
    return sortedArticles.filter((article) => {
      if (seen.has(article.id)) return false;
      seen.add(article.id);
      return true;
    });
  }, [sortedArticles]);

  const limit = 50;

  const displayedArticles = uniqueSortedArticles.slice(0, limit);
  const savedArticles = articles.filter((article) => savedIds.has(article.id)).slice(0, 6);

  return (
    <div className="dashboard">
      <AppHeader
        onRefresh={loadFeed}
        refreshing={refreshing}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        articleCount={articles.length}
        lastUpdated={lastUpdated}
      />
      <main className="page">
        {errorMessage && (
          <div className="notice error">
            <div style={{ flex: 1 }}>
              <strong>Ошибка подключения</strong>
              <p style={{ margin: "4px 0 0", fontSize: "14px", opacity: 0.9 }}>
                {errorMessage}
              </p>
              {errorMessage.includes("недоступен") || errorMessage.includes("подключиться") ? (
                <p style={{ margin: "8px 0 0", fontSize: "12px", opacity: 0.8 }}>
                  💡 Убедитесь, что бэкенд запущен: <code>cd backend && uvicorn app.main:app --reload</code>
                </p>
              ) : null}
            </div>
            <button onClick={loadFeed}>🔄 Повторить</button>
          </div>
        )}
        <section className="hero">
          <div>
            <h1>Persona News</h1>
            <p>
              Персональная лента с более чем 15 мировыми и региональными источниками. Лайкните то, что интересно, —
              рекомендации адаптируются мгновенно.
            </p>
            <ul className="hero-points">
              <li>🧠 ИИ-сводки по ключевым темам и персональным интересам</li>
              <li>⚡ Автообновление каждые 2 минуты + цвет свежести</li>
              <li>⭐ Избранное и скрытые материалы в одном клике</li>
            </ul>
          </div>
          <div className="hero-stats">
            <div>
              <span className="stat-label">Всего статей</span>
              <strong>{articles.length}</strong>
            </div>
            <div>
              <span className="stat-label">Тем</span>
              <strong>{topicStats.length}</strong>
            </div>
            <div>
              <span className="stat-label">Обновление</span>
              <strong>{lastUpdated ? "только что" : "каждые 2 мин"}</strong>
            </div>
          </div>
        </section>

        {!!savedArticles.length && (
          <section className="saved-ribbon" aria-label="Сохраненные статьи">
            <header>
              <div>
                <h3>Избранное</h3>
                <p>Подборка для второго чтения</p>
              </div>
              <span>{savedArticles.length} материалов</span>
            </header>
            <div className="saved-scroll">
              {savedArticles.map((article) => (
                <article key={article.id} className="saved-card">
                  <span className="saved-source">{article.source_name ?? "Источник"}</span>
                  <a href={article.url} target="_blank" rel="noreferrer">
                    {article.title}
                  </a>
                  <button onClick={() => handleToggleSave(article.id)}>Убрать</button>
                </article>
              ))}
            </div>
          </section>
        )}

        {topicStats.length > 0 && (
          <TopicFilter
            topics={topicStats}
            selectedTopics={selectedTopics}
            onToggleTopic={handleToggleTopic}
            onClearAll={handleClearFilter}
          />
        )}

        <FeedToolbar
          sortBy={sortBy}
          onSortChange={setSortBy}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          visibleCount={displayedArticles.length}
          totalCount={filteredArticles.length}
          savedCount={savedIds.size}
        />

        {loading ? (
          <p className="notice">Загрузка...</p>
        ) : filteredArticles.length === 0 ? (
          <p className="notice">
            {articles.length === 0
              ? "Нет данных: убедитесь, что backend запущен."
              : selectedTopics.size > 0
              ? "Нет статей по выбранным темам."
              : "Все статьи скрыты — верните их через панель снизу."}
          </p>
        ) : (
          <>
            <div className="feed-stats">
              Показано {displayedArticles.length} из {filteredArticles.length} статей
              {selectedTopics.size > 0 && ` • фильтр: ${Array.from(selectedTopics).join(", ")}`}
              {searchQuery && ` • поиск: "${searchQuery}"`}
            </div>
            <section className={`feed-grid ${viewMode === "list" ? "list-view" : ""}`}>
              {displayedArticles.map((article) => (
                <TopicCard
                  key={article.id}
                  article={article}
                  onHide={handleHideArticle}
                  viewMode={viewMode}
                  saved={savedIds.has(article.id)}
                  onToggleSave={handleToggleSave}
                />
              ))}
            </section>
          </>
        )}
        <HiddenPanel
          items={Object.values(hiddenRecords)}
          resolveTitle={resolveTitle}
          onRestore={handleRestoreArticle}
          currentTick={timerTick}
        />
      </main>
    </div>
  );
}
```

---

## `frontend/components/AppHeader.tsx`

```tsx
"use client";

type Props = {
  onRefresh?: () => void;
  refreshing?: boolean;
  searchQuery: string;
  onSearchChange: (value: string) => void;
  articleCount: number;
  lastUpdated: Date | null;
};

export function AppHeader({ onRefresh, refreshing, searchQuery, onSearchChange, articleCount, lastUpdated }: Props) {
  const formattedUpdated = lastUpdated
    ? new Intl.DateTimeFormat("ru-RU", {
        hour: "2-digit",
        minute: "2-digit"
      }).format(lastUpdated)
    : "—";

  return (
    <header className="app-header">
      <div className="logo-block">
        <div className="logo-dot" />
        <div>
          <span className="brand">Persona News</span>
          <p>Персональный ИИ-агрегатор</p>
        </div>
      </div>
      <div className="header-actions">
        <div className="header-meta">
          <span>В ленте {articleCount} материалов</span>
          <span className="dot" />
          <span>обновлено в {formattedUpdated}</span>
        </div>
        <input
          className="search-input"
          placeholder="Поиск по заголовку или источнику"
          type="search"
          value={searchQuery}
          onChange={(evt) => onSearchChange(evt.target.value)}
        />
        <div className="header-buttons">
          <button className="refresh-btn" onClick={onRefresh} disabled={refreshing}>
            ↻ {refreshing ? "Обновляем..." : "Обновить ленту"}
          </button>
        </div>
      </div>
    </header>
  );
}
```

---

## `frontend/components/FeedToolbar.tsx`

```tsx
"use client";

type SortOption = "home" | "fresh" | "source";
type ViewMode = "grid" | "list";

type Props = {
  sortBy: SortOption;
  onSortChange: (value: SortOption) => void;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  visibleCount: number;
  totalCount: number;
  savedCount: number;
};

const sortLabels: Record<SortOption, string> = {
  home: "Главная",
  fresh: "Сначала свежие",
 
};

export function FeedToolbar({
  sortBy,
  onSortChange,
  viewMode,
  onViewModeChange,
  visibleCount,
  totalCount,
  savedCount
}: Props) {
  return (
    <section className="feed-toolbar" aria-label="Панель управления лентой">
      <div className="toolbar-col stats">
        <span className="toolbar-metric">
          Показано <strong>{visibleCount}</strong> / {totalCount}
        </span>
        {savedCount > 0 && (
          <span className="saved-chip" title="Сохранено в быстрый доступ">
            ★ {savedCount} избранных
          </span>
        )}
      </div>
      <div className="toolbar-col controls">
        <div className="sort-switch">
          {Object.entries(sortLabels).map(([value, label]) => (
            <button
              key={value}
              className={`pill-button ${sortBy === value ? "active" : ""}`}
              onClick={() => onSortChange(value as SortOption)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="view-toggle" role="group" aria-label="Отображение">
          <button
            className={`toggle-btn ${viewMode === "grid" ? "active" : ""}`}
            onClick={() => onViewModeChange("grid")}
            aria-pressed={viewMode === "grid"}
          >
            ▦
          </button>
          <button
            className={`toggle-btn ${viewMode === "list" ? "active" : ""}`}
            onClick={() => onViewModeChange("list")}
            aria-pressed={viewMode === "list"}
          >
            ☰
          </button>
        </div>
      </div>
    </section>
  );
}
```

---

## `frontend/components/HiddenPanel.tsx`

```tsx
"use client";

import { HiddenRecord } from "../lib/api";

type Props = {
  items: HiddenRecord[];
  resolveTitle: (id: number) => string;
  onRestore: (id: number) => void;
  currentTick: number;
};

function formatCountdown(expiresAt: string, now: number): string {
  const diff = new Date(expiresAt).getTime() - now;
  if (diff <= 0) return "время истекло";
  const minutes = Math.floor(diff / 60000);
  const seconds = Math.floor((diff % 60000) / 1000)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${seconds}`;
}

export function HiddenPanel({ items, onRestore, resolveTitle, currentTick }: Props) {
  if (!items.length) return null;

  return (
    <div className="hidden-panel">
      <h4>Скрытые материалы (можно вернуть в течение часа)</h4>
      <ul>
        {items.map((item) => (
          <li key={item.article_id}>
            <div>
              <span className="hidden-title">{resolveTitle(item.article_id)}</span>
              <span className="hidden-timer">{formatCountdown(item.expires_at, currentTick)}</span>
            </div>
            <button onClick={() => onRestore(item.article_id)}>Вернуть</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

---

## `frontend/components/TopicCard.tsx`

```tsx
"use client";

import { Article, submitFeedback } from "../lib/api";
import { useMemo, useState } from "react";

type Props = {
  article: Article;
  onHide?: (payload: { id: number; title: string }) => void;
  viewMode?: "grid" | "list";
  saved?: boolean;
  onToggleSave?: (id: number) => void;
};

function formatTimeAgo(dateStr: string | null): string {
  if (!dateStr) return "Дата неизвестна";
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return "только что";
  if (diffMins < 60) return `${diffMins} мин. назад`;
  if (diffHours < 24) return `${diffHours} ч. назад`;
  return `${diffDays} дн. назад`;
}

function getFreshnessClass(dateStr: string | null): string {
  if (!dateStr) return "freshness-unknown";
  const date = new Date(dateStr);
  const now = new Date();
  const diffHours = (now.getTime() - date.getTime()) / 3600000;

  if (diffHours < 1) return "freshness-very-fresh";
  if (diffHours < 6) return "freshness-fresh";
  if (diffHours < 24) return "freshness-recent";
  return "freshness-old";
}

function estimateReadingTime(text: string): string {
  const words = text.split(/\s+/).length;
  const minutes = Math.max(1, Math.round(words / 160));
  return `${minutes} мин. чтения`;
}

function getSentimentBadge(value: number | null): { label: string; className: string } {
  if (value === null || value === undefined) return { label: "нейтрально", className: "sentiment-neutral" };
  if (value > 0.2) return { label: "позитив", className: "sentiment-positive" };
  if (value < -0.2) return { label: "негатив", className: "sentiment-negative" };
  return { label: "нейтрально", className: "sentiment-neutral" };
}

const TOPIC_BACKGROUND: Record<string, string> = {
  технологии: "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1400&q=80",
  technology: "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1400&q=80",
  экономика: "https://images.unsplash.com/photo-1454165205744-3b78555e5572?auto=format&fit=crop&w=1400&q=80",
  economy: "https://images.unsplash.com/photo-1454165205744-3b78555e5572?auto=format&fit=crop&w=1400&q=80",
  культура: "https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&w=1400&q=80",
  culture: "https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&w=1400&q=80",
  спорт: "https://images.unsplash.com/photo-1517649763962-0c623066013b?auto=format&fit=crop&w=1400&q=80",
  sport: "https://images.unsplash.com/photo-1517649763962-0c623066013b?auto=format&fit=crop&w=1400&q=80",
  политика: "https://images.unsplash.com/photo-1469474968028-56623f02e42e?auto=format&fit=crop&w=1400&q=80",
  politics: "https://images.unsplash.com/photo-1469474968028-56623f02e42e?auto=format&fit=crop&w=1400&q=80",
  наука: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1400&q=80",
  science: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1400&q=80",
  бизнес: "https://images.unsplash.com/photo-1434030216411-0b793f4b4173?auto=format&fit=crop&w=1400&q=80",
  business: "https://images.unsplash.com/photo-1434030216411-0b793f4b4173?auto=format&fit=crop&w=1400&q=80",
  медиа: "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1400&q=80",
  media: "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1400&q=80"
};

function getBackgroundImage(article: Article): string {
  if (article.image_url) return article.image_url;
  const topicMatch = article.topics.find((topic) => TOPIC_BACKGROUND[topic.toLowerCase()]);
  if (topicMatch) {
    return TOPIC_BACKGROUND[topicMatch.toLowerCase()];
  }
  return `https://source.unsplash.com/featured/900x600?${encodeURIComponent(article.topics[0] ?? "news")}`;
}

function stripHtml(value: string): string {
  return value.replace(/<\/?[^>]+(>|$)/g, "");
}

export function TopicCard({ article, onHide, viewMode = "grid", saved = false, onToggleSave }: Props) {
  const [feedbackSent, setFeedbackSent] = useState<string | null>(null);
  const [isHiding, setIsHiding] = useState(false);
  const previewImage = useMemo(() => getBackgroundImage(article), [article]);
  const summaryText = useMemo(() => {
    const cleaned = stripHtml(article.summary ?? "");
    return cleaned || "Описание скоро появится.";
  }, [article.summary]);
  const timeAgo = formatTimeAgo(article.published_at);
  const freshnessClass = getFreshnessClass(article.published_at);
  const sentiment = getSentimentBadge(article.sentiment);
  const readingTime = estimateReadingTime(summaryText);

  const handleFeedback = async (action: "like" | "dislike" | "hide") => {
    try {
      await submitFeedback(article.id, action);
      setFeedbackSent(action);
      if (action === "hide" && onHide) {
        setIsHiding(true);
        setTimeout(
          () =>
            onHide({
              id: article.id,
              title: article.title
            }),
          300
        );
      }
    } catch (error) {
      console.error("Ошибка отправки обратной связи:", error);
    }
  };

  if (isHiding) return null;

  return (
    <article className={`topic-card ${viewMode === "list" ? "list" : ""} ${saved ? "saved" : ""}`}>
      <div className="card-media" style={{ backgroundImage: `url("${previewImage}")` }}>
        <div className="media-overlay">
          <div className="media-top">
            <span className="source-pill">{article.source_name || "Источник"}</span>
            <span className={`freshness ${freshnessClass}`} title={article.published_at || undefined}>
              {timeAgo}
            </span>
          </div>
          <a className="media-title" href={article.url} target="_blank" rel="noreferrer">
            {article.title}
          </a>
          <div className="topic-tags">
            {article.topics.slice(0, 3).map((topic) => (
              <span key={topic} className="topic-badge">
                {topic}
              </span>
            ))}
          </div>
        </div>
      </div>
      <div className="card-body">
        <div className="card-meta">
          <span>{article.source_name || "Источник"}</span>
          <span className={`sentiment ${sentiment.className}`}>{sentiment.label}</span>
          <span>{readingTime}</span>
        </div>
        <p>{summaryText}</p>
      </div>
      <footer>
        <div className="entities">
          <strong>Сущности:</strong> {article.entities.join(", ") || "нет"}
        </div>
        <div className="feedback-actions">
          {onToggleSave && (
            <button
              className={`feedback-btn bookmark ${saved ? "active" : ""}`}
              onClick={() => onToggleSave(article.id)}
              title={saved ? "Удалить из избранного" : "Сохранить"}
              aria-pressed={saved}
            >
              {saved ? "★" : "☆"}
            </button>
          )}
          <button
            className={`feedback-btn like ${feedbackSent === "like" ? "active" : ""}`}
            onClick={() => handleFeedback("like")}
            title="Нравится"
          >
            👍
          </button>
          <button
            className={`feedback-btn dislike ${feedbackSent === "dislike" ? "active" : ""}`}
            onClick={() => handleFeedback("dislike")}
            title="Не нравится"
          >
            👎
          </button>
          <button className="feedback-btn hide" onClick={() => handleFeedback("hide")} title="Скрыть">
            ✕
          </button>
        </div>
      </footer>
    </article>
  );
}
```

---

## `frontend/components/TopicFilter.tsx`

```tsx
"use client";

type TopicInfo = {
  name: string;
  count: number;
};

type Props = {
  topics: TopicInfo[];
  selectedTopics: Set<string>;
  onToggleTopic: (topic: string) => void;
  onClearAll: () => void;
};

const ALL_TOPICS = [
  "Общее",
  "Технологии",
  "Кибербезопасность",
  "Экономика",
  "Общество",
  "Культура",
  "Наука",
  "Спорт",
];

export function TopicFilter({ topics, selectedTopics, onToggleTopic, onClearAll }: Props) {
  const topicCountMap = new Map(topics.map((t) => [t.name, t.count]));
  
  const allTopicsWithCounts = ALL_TOPICS.map((name) => ({
    name,
    count: topicCountMap.get(name) ?? 0
  }));

  return (
    <div className="topic-filter">
      <div className="filter-header">
        <h3>Темы</h3>
        {selectedTopics.size > 0 && (
          <button className="clear-filter" onClick={onClearAll}>
            Сбросить
          </button>
        )}
      </div>
      <div className="filter-tags">
        {allTopicsWithCounts.map((topic) => (
          <button
            key={topic.name}
            className={`filter-tag ${selectedTopics.has(topic.name) ? "active" : ""}`}
            onClick={() => onToggleTopic(topic.name)}
          >
            {topic.name}
          </button>
        ))}
      </div>
    </div>
  );
}
```

---

## `frontend/lib/api.ts`

```typescript
const DEFAULT_API_BASE = "http://localhost:8000";

const normalizeBaseUrl = (raw?: string | null): string => {
  if (!raw) return DEFAULT_API_BASE;
  let value = raw.trim();
  if (!value) return DEFAULT_API_BASE;
  if (!/^https?:\/\//i.test(value)) {
    value = `http://${value}`;
  }
  return value.replace(/\/+$/, "");
};

const API_BASE_URL = normalizeBaseUrl(process.env.NEXT_PUBLIC_API_BASE ?? null);

export type Article = {
  id: number;
  title: string;
  summary: string;
  url: string;
  topics: string[];
  entities: string[];
  published_at: string | null;
  created_at?: string | null;
  sentiment: number | null;
  source_name?: string | null;
  image_url?: string | null;
};

export type HiddenRecord = {
  article_id: number;
  hidden_at: string;
  expires_at: string;
};

export type FeedResponse = {
  articles: Article[];
  hidden: HiddenRecord[];
};

async function apiRequest(endpoint: string, options: RequestInit = {}): Promise<Response> {
  const path = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  const url = `${API_BASE_URL}${path}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });
    return res;
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new Error("Не удалось подключиться к серверу. Убедитесь, что бэкенд запущен на порту 8000.");
    }
    throw error;
  }
}

export async function fetchFeed(limit: number = 40): Promise<FeedResponse> {
  const searchParams = new URLSearchParams({
    user_id: "demo",
    limit: String(limit)
  });

  const res = await apiRequest(`/feed?${searchParams.toString()}`, {
    cache: "no-store"
  });
  
  if (!res.ok) {
    if (res.status === 0 || res.status >= 500) {
      throw new Error("Сервер недоступен. Проверьте, что бэкенд запущен.");
    }
    throw new Error(`Ошибка запроса: ${res.statusText}`);
  }
  
  return res.json();
}

export async function fetchSources(): Promise<any[]> {
  const res = await apiRequest("/sources");
  if (!res.ok) {
    return [];
  }
  return res.json();
}

export async function submitFeedback(
  articleId: number,
  action: "like" | "dislike" | "hide" | "undo_hide",
  userId: string = "demo"
): Promise<void> {
  const res = await apiRequest(
    `/feedback?article_id=${articleId}&user_id=${userId}&action=${action}`,
    { method: "POST" }
  );
  if (!res.ok) {
    throw new Error("Ошибка отправки обратной связи");
  }
}
```

---

## `start-dev.bat`

```batch
@echo off
setlocal enabledelayedexpansion

REM Resolve important paths
set "SCRIPT_DIR=%~dp0"
set "BACKEND_DIR=%SCRIPT_DIR%backend"
set "FRONTEND_DIR=%SCRIPT_DIR%frontend"
set "VENV_DIR=%BACKEND_DIR%\.venv"

echo ================================================
echo   Persona News - Dev Environment Bootstrap
echo ================================================
echo.

REM Ensure backend virtual environment exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
  echo [backend] Creating virtual environment...
  pushd "%BACKEND_DIR%"
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment. Ensure Python is installed.
    pause
    exit /b 1
  )
  call ".venv\Scripts\activate.bat"
  echo [backend] Installing dependencies...
  pip install -r requirements.txt
  call ".venv\Scripts\deactivate.bat"
  popd
) else (
  echo [backend] Using existing virtual environment.
)

REM Ensure frontend dependencies exist
if not exist "%FRONTEND_DIR%\node_modules" (
  echo [frontend] Installing npm dependencies...
  pushd "%FRONTEND_DIR%"
  npm install
  popd
) else (
  echo [frontend] node_modules already present.
)

echo.
echo Launching services...

REM Start backend server
start "Persona Backend" cmd /k "cd /d ""%BACKEND_DIR%"" && call ""%VENV_DIR%\Scripts\activate.bat"" && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Small delay before starting frontend
timeout /t 2 /nobreak >nul

REM Start frontend
start "Persona Frontend" cmd /k "cd /d ""%FRONTEND_DIR%"" && set NEXT_PUBLIC_API_BASE=http://localhost:8000 && npm run dev"

REM Verify backend health endpoint
echo.
echo Checking backend availability...
for /l %%I in (1,1,15) do (
  powershell -Command "try {Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:8000/health' -TimeoutSec 2 ^| Out-Null; exit 0} catch {exit 1}"
  if not errorlevel 1 (
    echo Backend is responding on http://localhost:8000
    goto done_check
  )
  timeout /t 1 /nobreak >nul >nul
)
echo WARNING: Backend didn't respond yet. Check the ""Persona Backend"" window for errors.

:done_check
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Both servers are now starting in separate windows.
echo This window can be closed; services keep running.
pause
```

---

## `scripts/md_to_docx.py`

```python
from docx import Document
from pathlib import Path
import re

md_path = Path('docs/PROJECT_DOCUMENTATION.md')
output_path = Path('docs/PROJECT_DOCUMENTATION.docx')

def process_line(doc, line):
    stripped = line.rstrip()
    if not stripped:
        doc.add_paragraph('')
        return
    heading_match = re.match(r'^(#+)\s+(.*)$', stripped)
    if heading_match:
        level = min(len(heading_match.group(1)), 4)
        text = heading_match.group(2).strip()
        doc.add_heading(text, level=level)
        return
    if stripped.startswith(('- ', '* ')):
        doc.add_paragraph(stripped[2:].strip(), style='List Bullet')
        return
    if re.match(r'^\d+\.\s+', stripped):
        doc.add_paragraph(stripped, style='List Number')
        return
    doc.add_paragraph(stripped)

def main():
    if not md_path.exists():
        raise SystemExit('Markdown file not found')
    doc = Document()
    for line in md_path.read_text(encoding='utf-8').splitlines():
        process_line(doc, line)
    doc.save(output_path)
    print(f'Generated {output_path}')

if __name__ == '__main__':
    main()
```

---

_Конец листинга кода_

