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
    sources: List[RssSource] = field(default_factory=lambda: DEFAULT_SOURCES[:15])  # Используем 15 источников (больше русских)
    ttl_minutes: int = 30  # Увеличиваем время кэша
    limit_per_source: int = 15  # Уменьшаем количество статей на источник
    _articles: List[Article] = field(default_factory=list)
    _last_updated: datetime | None = None
    _is_fetching: bool = False

    def is_expired(self) -> bool:
        if not self._last_updated:
            return True
        return datetime.utcnow() - self._last_updated > timedelta(minutes=self.ttl_minutes)

    def get_articles(self) -> List[Article]:
        # Если кэш истек и мы не загружаем данные сейчас, запускаем обновление
        if self.is_expired() and not self._is_fetching:
            # Если данных нет вообще, загружаем синхронно (первый запуск)
            if not self._articles:
                try:
                    self._is_fetching = True
                    # Загружаем только первые 3 источника для быстрой первой загрузки
                    quick_sources = self.sources[:3]
                    quick_articles = []
                    for source_id, source in enumerate(quick_sources, start=1):
                        try:
                            quick_articles.extend(self._fetch_source(source_id, source))
                        except:
                            pass
                    self._articles = quick_articles[:50]  # Первые 50 статей быстро
                    self._last_updated = datetime.utcnow()
                    self._is_fetching = False
                    
                    # Затем загружаем остальные в фоне
                    import threading
                    def update_cache_full():
                        try:
                            self._is_fetching = True
                            new_articles = self._fetch()
                            self._articles = new_articles
                            self._last_updated = datetime.utcnow()
                        except Exception as e:
                            print(f"Ошибка обновления кэша: {e}")
                        finally:
                            self._is_fetching = False
                    thread = threading.Thread(target=update_cache_full, daemon=True)
                    thread.start()
                except Exception as e:
                    print(f"Ошибка быстрой загрузки: {e}")
                    self._is_fetching = False
            else:
                # Если есть старые данные, обновляем в фоне
                import threading
                def update_cache():
                    try:
                        self._is_fetching = True
                        new_articles = self._fetch()
                        self._articles = new_articles
                        self._last_updated = datetime.utcnow()
                    except Exception as e:
                        print(f"Ошибка обновления кэша: {e}")
                    finally:
                        self._is_fetching = False
                thread = threading.Thread(target=update_cache, daemon=True)
                thread.start()
        # Всегда возвращаем текущие данные
        return self._articles

    def _fetch_source(self, source_id: int, source: RssSource) -> List[Article]:
        batch: List[Article] = []
        try:
            for entry in fetch(source, limit=self.limit_per_source):
                try:
                    url = entry.get("link")
                    if not url:
                        continue
                    title = entry.get("title") or "Без названия"
                    raw_summary = entry.get("summary") or title
                    # Упрощаем обработку для скорости - используем оригинальные тексты без перевода
                    summary = summarizer.summarize(raw_summary)
                    published_at = _parse_datetime(entry)
                    topics = classify(f"{title} {raw_summary}")
                    # Переводим на русский, но только если текст не на русском
                    # Проверяем, есть ли кириллица в тексте
                    has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in title)
                    if has_cyrillic:
                        # Уже на русском, не переводим
                        ru_title = title
                        ru_summary = summary
                    else:
                        # Переводим на русский
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
                        entities=_extract_entities(title + " " + summary),
                        sentiment=None,
                        published_at=published_at,
                        created_at=datetime.utcnow(),
                        image_url=_extract_image(entry),
                    )
                    batch.append(article)
                except Exception as e:
                    # Пропускаем проблемную статью, продолжаем обработку остальных
                    print(f"Ошибка обработки статьи из {source.name}: {e}")
                    continue
        except Exception as e:
            print(f"Ошибка загрузки источника {source.name}: {e}")
        return batch

    def _fetch(self) -> List[Article]:
        collected: dict[str, Article] = {}
        # Уменьшаем количество потоков для стабильности
        with ThreadPoolExecutor(max_workers=min(5, len(self.sources))) as executor:
            futures = [
                executor.submit(self._fetch_source, source_id, source)
                for source_id, source in enumerate(self.sources, start=1)
            ]
            for future in as_completed(futures):
                try:
                    # Таймаут на каждый источник
                    result = future.result(timeout=15)
                    for article in result:
                        if article.url not in collected:
                            collected[article.url] = article
                except Exception as e:
                    # Если источник не загрузился, просто пропускаем его
                    print(f"Ошибка загрузки источника: {e}")
                    continue
        ordered = sorted(
            collected.values(),
            key=lambda art: art.published_at or datetime.utcnow(),
            reverse=True,
        )
        return ordered[:100]  # Уменьшаем до 100 статей для скорости


feed_cache = FeedCache()

