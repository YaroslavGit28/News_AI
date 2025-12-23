from functools import lru_cache
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()


class Settings(BaseModel):
    app_name: str = Field(default="Persona News API")
    environment: str = Field(default=os.getenv("ENV", "development"))
    database_url: str = Field(default=os.getenv("DATABASE_URL", "sqlite:///./news.db"))
    redis_url: str = Field(default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    openai_api_key: str | None = Field(default=os.getenv("OPENAI_API_KEY"))
    deepseek_api_key: str | None = Field(default=os.getenv("DEEPSEEK_API_KEY"))
    deepseek_api_url: str = Field(default=os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
