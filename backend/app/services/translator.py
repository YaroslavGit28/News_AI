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

