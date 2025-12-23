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
