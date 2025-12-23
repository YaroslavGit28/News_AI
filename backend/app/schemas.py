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


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    message: str
