"""Настройка подключения к базе данных"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from .config import get_settings
from .models import Base

settings = get_settings()

# Создаем движок базы данных
# Для SQLite используем StaticPool для совместимости с async
if settings.database_url.startswith("sqlite"):
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
else:
    engine = create_engine(settings.database_url, echo=False)

# Создаем фабрику сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Инициализирует базу данных - создает все таблицы"""
    Base.metadata.create_all(bind=engine)
    print("База данных инициализирована")


def get_db() -> Session:
    """Dependency для получения сессии БД"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Получить сессию БД напрямую"""
    return SessionLocal()

