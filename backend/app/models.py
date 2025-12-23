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
