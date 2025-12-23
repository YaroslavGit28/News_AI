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
