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
    try:
        import urllib.request
        import socket
        
        # Устанавливаем таймаут для запроса
        socket.setdefaulttimeout(10)  # 10 секунд на источник
        
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
    except Exception as e:
        print(f"Ошибка загрузки источника {source.name}: {e}")
        # Возвращаем пустой список при ошибке
        return


DEFAULT_SOURCES: List[RssSource] = [
    # Русские источники (приоритет)
    RssSource(name="РИА Новости", url="https://ria.ru/export/rss2/index.xml", category="world", reliability=0.9),
    RssSource(name="ТАСС", url="https://tass.ru/rss/v2.xml", category="world", reliability=0.92),
    RssSource(name="Интерфакс", url="https://www.interfax.ru/rss.asp", category="world", reliability=0.88),
    RssSource(name="Lenta.ru", url="https://lenta.ru/rss/news", category="society", reliability=0.85),
    RssSource(name="Meduza", url="https://meduza.io/rss/all", category="society", reliability=0.87),
    RssSource(name="RBC", url="https://rssexport.rbc.ru/rbcnews/news/20/full.rss", category="business", reliability=0.86),
    RssSource(name="Ведомости", url="https://www.vedomosti.ru/rss/news", category="business", reliability=0.85),
    RssSource(name="Коммерсант", url="https://www.kommersant.ru/RSS/news.xml", category="business", reliability=0.88),
    RssSource(name="Газета.ru", url="https://www.gazeta.ru/export/rss/lenta.xml", category="society", reliability=0.84),
    RssSource(name="RT на русском", url="https://russian.rt.com/rss", category="world", reliability=0.82),
    RssSource(name="Deutsche Welle", url="https://rss.dw.com/russian", category="society", reliability=0.88),
    RssSource(name="N+1", url="https://nplus1.ru/rss", category="science", reliability=0.85),
    RssSource(name="Хабр", url="https://habr.com/ru/rss/all/all/", category="technology", reliability=0.9),
    RssSource(name="VC.ru", url="https://vc.ru/rss", category="business", reliability=0.87),
    RssSource(name="3DNews", url="https://3dnews.ru/breaking/rss", category="technology", reliability=0.83),
    
    # Международные источники (для разнообразия)
    RssSource(name="BBC World", url="https://feeds.bbci.co.uk/news/world/rss.xml", category="world", reliability=0.95),
    RssSource(name="Reuters World", url="https://feeds.reuters.com/Reuters/worldNews", category="world", reliability=0.93),
    RssSource(name="The Guardian", url="https://www.theguardian.com/world/rss", category="world", reliability=0.9),
    RssSource(name="TechCrunch", url="https://techcrunch.com/feed/", category="technology", reliability=0.9),
    RssSource(name="The Verge", url="https://www.theverge.com/rss/index.xml", category="technology", reliability=0.9),
]
