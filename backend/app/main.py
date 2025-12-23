from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .config import get_settings
from .datasources.rss import DEFAULT_SOURCES
from .schemas import HiddenArticleInfo, RecommendationResponse, Source, HealthResponse, ChatRequest, ChatResponse, Article
from .services.feed_cache import FeedCache, feed_cache
from .services.recommender import SimpleRecommender
from .services.deepseek_client import call_deepseek_api
from .tasks.ingest import ingest_sources
from .database import init_db, get_db, get_db_session
from .models import User, UserFeedback

settings = get_settings()
app = FastAPI(title="Persona News API")
recommender = SimpleRecommender()
cache: FeedCache = feed_cache

# Инициализируем БД при старте приложения
@app.on_event("startup")
def startup_event():
    init_db()

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
    db: Session = Depends(get_db),
) -> RecommendationResponse:
    # Создаем пользователя, если его нет
    get_or_create_user(db, user_id)
    
    articles = cache.get_articles()
    hidden = _cleanup_hidden(db, user_id)
    visible_articles = [article for article in articles if article.id not in hidden]
    personalized = recommender.recommend(user_id=user_id, articles=visible_articles)
    
    # Получаем информацию о скрытых статьях из БД
    hidden_feedbacks = db.query(UserFeedback).filter(
        UserFeedback.user_id == user_id,
        UserFeedback.action == "hide"
    ).all()
    
    hidden_info = [
        HiddenArticleInfo(
            article_id=fb.article_id,
            hidden_at=fb.created_at,
            expires_at=fb.expires_at or (fb.created_at + timedelta(hours=1)),
        )
        for fb in hidden_feedbacks
        if not fb.expires_at or fb.expires_at > datetime.utcnow()
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
    db: Session = Depends(get_db),
) -> dict:
    """Обратная связь от пользователя для улучшения рекомендаций"""
    if action not in ["like", "dislike", "hide", "undo_hide"]:
        raise HTTPException(status_code=400, detail="Invalid action. Use: like, dislike, hide, undo_hide")

    # Создаем пользователя, если его нет
    user = get_or_create_user(db, user_id)
    article = _get_article(article_id)

    if action in {"like", "dislike"} and not article:
        raise HTTPException(status_code=404, detail="Article not found for feedback")

    message = "ok"
    
    if action == "like":
        # Удаляем старые feedback для этой статьи
        db.query(UserFeedback).filter(
            UserFeedback.user_id == user_id,
            UserFeedback.article_id == article_id
        ).delete()
        
        # Добавляем новый like
        feedback = UserFeedback(
            user_id=user_id,
            article_id=article_id,
            action="like"
        )
        db.add(feedback)
        recommender.add_feedback(user_id, article, value=1)
        message = "liked"
        
    elif action == "dislike":
        # Удаляем старые feedback для этой статьи
        db.query(UserFeedback).filter(
            UserFeedback.user_id == user_id,
            UserFeedback.article_id == article_id
        ).delete()
        
        # Добавляем новый dislike
        feedback = UserFeedback(
            user_id=user_id,
            article_id=article_id,
            action="dislike"
        )
        db.add(feedback)
        recommender.add_feedback(user_id, article, value=-1)
        message = "disliked"
        
    elif action == "hide":
        # Проверяем, не скрыта ли уже статья
        existing = db.query(UserFeedback).filter(
            UserFeedback.user_id == user_id,
            UserFeedback.article_id == article_id,
            UserFeedback.action == "hide"
        ).first()
        
        if not existing:
            feedback = UserFeedback(
                user_id=user_id,
                article_id=article_id,
                action="hide",
                expires_at=datetime.utcnow() + timedelta(hours=1)
            )
            db.add(feedback)
        message = "hidden"
        
    elif action == "undo_hide":
        # Удаляем скрытие
        removed = db.query(UserFeedback).filter(
            UserFeedback.user_id == user_id,
            UserFeedback.article_id == article_id,
            UserFeedback.action == "hide"
        ).delete()
        message = "restored" if removed > 0 else "not_hidden"

    db.commit()
    return {"status": "success", "message": message, "user_id": user_id}


def _generate_ai_response(user_message: str, articles: list) -> str:
    """Генерирует ответ AI ассистента на основе сообщения пользователя и доступных новостей"""
    message_lower = user_message.lower()
    
    # Анализ новостей
    if any(word in message_lower for word in ["новости", "что происходит", "тренды", "события", "анализ"]):
        if not articles:
            return "К сожалению, сейчас нет доступных новостей для анализа. Попробуйте позже, когда лента обновится."
        
        topics = {}
        for article in articles[:20]:
            for topic in article.topics:
                topics[topic] = topics.get(topic, 0) + 1
        
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
        topic_list = ", ".join([f"{topic} ({count})" for topic, count in top_topics])
        
        recent_count = 0
        for a in articles[:20]:
            if a.published_at:
                try:
                    if isinstance(a.published_at, str):
                        pub_date = datetime.fromisoformat(a.published_at.replace('Z', '+00:00')).replace(tzinfo=None)
                    else:
                        pub_date = a.published_at
                    if (datetime.utcnow() - pub_date).days < 1:
                        recent_count += 1
                except (ValueError, AttributeError):
                    pass
        
        response = f"📊 Отличный вопрос! Давайте разберемся, что происходит в мире новостей прямо сейчас.\n\n"
        response += f"В нашей ленте сейчас **{len(articles)} статей** из различных источников. "
        response += f"Из них **{recent_count} новостей** были опубликованы за последние 24 часа - это довольно свежий контент!\n\n"
        response += f"**Топ-5 самых популярных тем** в текущей ленте:\n"
        for i, (topic, count) in enumerate(top_topics, 1):
            response += f"{i}. {topic} ({count} статей)\n"
        response += f"\nИнтересно, что {top_topics[0][0] if top_topics else 'новости'} доминирует в ленте - это говорит о том, что эта тема сейчас очень актуальна.\n\n"
        response += "Хотите узнать больше о какой-то конкретной теме? Или может быть, вас интересуют новости из определенного источника? Просто спросите!"
        return response
    
    # Обсуждение будущего
    if any(word in message_lower for word in ["будущее", "прогноз", "что будет", "предсказание", "тренды будущего"]):
        response = "🔮 О, это очень интересная тема! Давайте поразмышляем о будущем на основе текущих трендов.\n\n"
        
        # Анализируем текущие темы в новостях
        if articles:
            topics = {}
            for article in articles[:30]:
                for topic in article.topics:
                    topics[topic] = topics.get(topic, 0) + 1
            top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
            
            response += "Исходя из того, что сейчас доминирует в новостной ленте, я вижу несколько ключевых направлений:\n\n"
            
            if any("технолог" in t[0].lower() for t in top_topics):
                response += "**🤖 Технологии и AI:**\n"
                response += "Развитие искусственного интеллекта, автоматизации и цифровизации продолжается ускоренными темпами. "
                response += "Эти технологии уже сейчас трансформируют многие отрасли, и в будущем их влияние только усилится. "
                response += "Ожидаю, что мы увидим больше интеграции AI в повседневную жизнь, новые прорывы в робототехнике и автоматизации процессов.\n\n"
            
            if any("эконом" in t[0].lower() for t in top_topics):
                response += "**💰 Экономика и финансы:**\n"
                response += "Экономические тренды показывают, что мир продолжает адаптироваться к новым реалиям. "
                response += "Цифровая экономика, удаленная работа, новые бизнес-модели - все это становится нормой. "
                response += "В будущем, вероятно, мы увидим еще больше инноваций в финансовых технологиях и способах ведения бизнеса.\n\n"
            
            response += "**🌍 Глобальные вызовы:**\n"
            response += "Климатические изменения, геополитические сдвиги, социальные трансформации - все это будет определять повестку ближайших лет. "
            response += "Страны и компании все больше фокусируются на устойчивом развитии и адаптации к новым условиям.\n\n"
        else:
            response += "Исходя из общих трендов, можно выделить несколько направлений:\n\n"
            response += "• **Технологии**: Продолжается развитие AI, автоматизации и цифровизации. Эти области будут определять экономику будущего.\n"
            response += "• **Экология**: Климатические изменения и устойчивое развитие становятся приоритетами для многих стран.\n"
            response += "• **Общество**: Цифровая трансформация меняет способы работы, общения и потребления информации.\n\n"
        
        response += "Какое направление вас больше всего интересует? Могу проанализировать конкретные новости по этой теме или обсудить детали подробнее!"
        return response
    
    # Аналитика
    if any(word in message_lower for word in ["анализ", "аналитика", "статистика", "данные", "исследование"]):
        if not articles:
            return "Для анализа нужны новости. Подождите, пока лента обновится."
        
        sources = {}
        sentiments = []
        for article in articles[:30]:
            sources[article.source_name or "Неизвестно"] = sources.get(article.source_name or "Неизвестно", 0) + 1
            if article.sentiment is not None:
                sentiments.append(article.sentiment)
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        sentiment_label = "позитивный" if avg_sentiment > 0.1 else "негативный" if avg_sentiment < -0.1 else "нейтральный"
        
        top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:3]
        sources_list = ", ".join([f"{name} ({count})" for name, count in top_sources])
        
        response = "📈 Отличный запрос! Давайте посмотрим на аналитику новостной ленты.\n\n"
        response += f"Общая статистика:\n"
        response += f"• Всего источников в ленте: {len(sources)}\n"
        response += f"• Проанализировано статей: {min(30, len(articles))}\n\n"
        
        if top_sources:
            response += f"Топ-3 самых активных источника:\n"
            for i, (name, count) in enumerate(top_sources, 1):
                response += f"{i}. {name} - {count} статей\n"
            response += "\n"
        
        if sentiments:
            response += f"Тональность новостей:\n"
            response += f"Средний тон новостей: {sentiment_label} (индекс: {avg_sentiment:.2f})\n"
            if avg_sentiment > 0.1:
                response += "Это говорит о том, что в ленте преобладают позитивные новости. "
            elif avg_sentiment < -0.1:
                response += "В ленте больше негативных новостей, что может отражать сложную ситуацию. "
            else:
                response += "Тональность в целом нейтральная, что характерно для информационных новостей. "
            response += "\n\n"
        
        response += "Эти данные дают хорошее представление о текущем информационном потоке. "
        response += "Хотите узнать больше о какой-то конкретной теме или источнике? Или может быть, вас интересует более детальный анализ?"
        return response
    
    # Общие вопросы
    if any(word in message_lower for word in ["привет", "здравствуй", "hello", "hi"]):
        article_count = len(articles)
        return (f"Привет! 👋 Очень рад вас видеть!\n\n"
                f"Я ваш персональный AI-ассистент для новостей. Сейчас в моей базе {article_count} статей "
                f"из различных источников, и я готов помочь вам разобраться в том, что происходит в мире.\n\n"
                f"Что я умею:**\n"
                f"• 📰 Обсуждать текущие новости и анализировать тренды\n"
                f"• 📊 Предоставлять детальную аналитику по источникам и темам\n"
                f"• 🔮 Размышлять о будущих трендах на основе текущих данных\n"
                f"• 💬 Отвечать на ваши вопросы о новостях и их контексте\n"
                f"• 🔍 Искать статьи по интересующим вас темам\n\n"
                f"О чем бы вы хотели поговорить? Можете спросить что-то вроде 'что происходит в мире?', "
                f"'расскажи о технологиях' или 'какие новости об экономике?' - я с удовольствием помогу!")
    
    if any(word in message_lower for word in ["помощь", "help", "что ты умеешь", "возможности"]):
        return ("🤖 Отлично, что спросили! Позвольте рассказать о моих возможностях подробнее:\n\n"
                "📰 Обсуждение новостей:\n"
                "Я могу рассказать о текущих трендах и событиях, проанализировать, что происходит в разных сферах жизни, "
                "и помочь вам понять контекст тех или иных новостей.\n\n"
                "📊 Аналитика и статистика:\n"
                "Могу предоставить детальную статистику по источникам, темам, тональности новостей. "
                "Например, сколько новостей из какой темы, какие источники наиболее активны, и так далее.\n\n"
                "🔮 Прогнозы и тренды:\n"
                "На основе текущих новостей могу обсудить возможные будущие тренды и направления развития событий.\n\n"
                "💬 Консультации и поиск:\n"
                "Отвечаю на ваши вопросы о новостях, могу найти статьи по интересующим темам, "
                "объяснить контекст событий.\n\n"
                "**Примеры вопросов, которые вы можете задать:\n"
                "• 'Что происходит в мире?'\n"
                "• 'Расскажи о новостях в технологиях'\n"
                "• 'Какая статистика по источникам?'\n"
                "• 'Что будет с экономикой?'\n"
                "• 'Найди новости про AI'\n\n"
                "Просто напишите ваш вопрос естественным языком - я постараюсь помочь!")
    
    # Поиск по темам
    if any(word in message_lower for word in ["технологии", "тех", "tech", "ai", "искусственный интеллект"]):
        tech_articles = [a for a in articles if "Технологии" in a.topics or any("тех" in t.lower() for t in a.topics)]
        if tech_articles:
            tech_topics = set()
            sources_tech = {}
            for a in tech_articles[:15]:
                tech_topics.update(a.topics)
                sources_tech[a.source_name or "Неизвестно"] = sources_tech.get(a.source_name or "Неизвестно", 0) + 1
            topics_list = ", ".join(list(tech_topics)[:6])
            top_source = max(sources_tech.items(), key=lambda x: x[1])[0] if sources_tech else None
            
            response = f"💻 Отлично! Технологии - это одна из самых динамичных областей!\n\n"
            response += f"В ленте сейчас **{len(tech_articles)} статей** о технологиях. "
            response += f"Это довольно много, что говорит о том, что эта тема очень актуальна!\n\n"
            
            if topics_list:
                response += f"Основные темы в технологических новостях:\n{topics_list}\n\n"
            
            if top_source:
                response += f"Больше всего новостей о технологиях публикует **{top_source}** - они явно активно следят за этой темой.\n\n"
            
            response += "Хотите узнать больше о какой-то конкретной новости? Или может быть, вас интересует определенная технология? "
            response += "Просто спросите, и я найду релевантные статьи!"
            return response
        return "К сожалению, в текущей ленте пока нет новостей о технологиях. Но не расстраивайтесь - попробуйте обновить ленту через несколько минут, и новые статьи обязательно появятся!"
    
    if any(word in message_lower for word in ["экономика", "экономи", "economy", "финансы", "рынок", "бирж"]):
        econ_articles = [a for a in articles if "Экономика" in a.topics]
        if econ_articles:
            sources = {}
            recent_econ = 0
            for a in econ_articles[:20]:
                sources[a.source_name or "Неизвестно"] = sources.get(a.source_name or "Неизвестно", 0) + 1
                if a.published_at:
                    try:
                        if isinstance(a.published_at, str):
                            pub_date = datetime.fromisoformat(a.published_at.replace('Z', '+00:00')).replace(tzinfo=None)
                        else:
                            pub_date = a.published_at
                        if (datetime.utcnow() - pub_date).days < 1:
                            recent_econ += 1
                    except:
                        pass
            
            top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:3]
            
            response = f"💰 Экономика - это всегда интересная тема! Давайте посмотрим, что происходит.\n\n"
            response += f"В ленте сейчас **{len(econ_articles)} статей** об экономике. "
            if recent_econ > 0:
                response += f"Из них **{recent_econ} новостей** были опубликованы за последние 24 часа - это свежие данные!\n\n"
            else:
                response += "\n\n"
            
            if top_sources:
                response += f"Основные источники экономических новостей:\n"
                for i, (name, count) in enumerate(top_sources, 1):
                    response += f"{i}. {name} - {count} статей\n"
                response += "\n"
            
            response += "Текущие темы в экономических новостях:** рынки, инвестиции, бизнес, финансы, валюта, инфляция.\n\n"
            response += "Что именно вас интересует? Могу найти конкретные новости о рынках, инвестициях, или может быть, вас интересует что-то другое?"
            return response
        return "К сожалению, в ленте пока нет экономических новостей. Попробуйте обновить ленту через несколько минут - новые статьи обязательно появятся!"
    
    # Поиск по конкретным словам в новостях
    if articles:
        query_words = [w for w in message_lower.split() if len(w) > 3]
        matching_articles = []
        for article in articles[:50]:
            article_text = f"{article.title} {article.summary} {' '.join(article.topics)} {' '.join(article.entities)}".lower()
            if any(word in article_text for word in query_words):
                matching_articles.append(article)
        
        if matching_articles:
            return (f"🔍 Найдено {len(matching_articles)} статей, связанных с вашим запросом:\n\n"
                    f"• {matching_articles[0].title}\n"
                    f"  {matching_articles[0].summary[:100]}...\n"
                    f"  Источник: {matching_articles[0].source_name}\n\n"
                    "Хотите узнать больше о какой-то конкретной новости?")
    
    # Дефолтный ответ с более полезной информацией
    if articles:
        # Пытаемся найти релевантные статьи
        query_words = [w for w in message_lower.split() if len(w) > 3]
        matching_articles = []
        for article in articles[:50]:
            article_text = f"{article.title} {article.summary} {' '.join(article.topics)} {' '.join(article.entities)}".lower()
            if any(word in article_text for word in query_words):
                matching_articles.append(article)
        
        if matching_articles:
            response = f"Отлично! Я нашел {len(matching_articles)} статей**, связанных с вашим запросом '{user_message}'.\n\n"
            response += f"Вот несколько примеров:\n\n"
            for i, article in enumerate(matching_articles[:3], 1):
                response += f"{i}. {article.title}\n"
                if article.summary:
                    summary_short = article.summary[:100] + "..." if len(article.summary) > 100 else article.summary
                    response += f"   {summary_short}\n"
                response += f"   Источник: {article.source_name}\n\n"
            response += "Хотите узнать больше о какой-то конкретной новости? Или может быть, вас интересует другая тема?"
            return response
        
        return (f"Спасибо за ваш вопрос о '{user_message}'! 😊\n\n"
                f"В текущей ленте у меня есть {len(articles)} новостей из различных источников. "
                f"Я могу помочь вам разобраться в том, что происходит в мире!\n\n"
                "Вот что я могу сделать:\n"
                "• 📰 Анализ новостей - спросите 'что происходит' или 'какие тренды'\n"
                "• 🔍 Поиск по темам - упомяните интересующую тему (технологии, экономика, спорт и т.д.)\n"
                "• 📊 Аналитика - спросите 'статистика' или 'анализ данных'\n"
                "• 🔮 Прогнозы - спросите о будущих трендах\n"
                "• 💬 Обсуждение - задайте любой вопрос о новостях\n\n"
                "Попробуйте задать более конкретный вопрос, например:\n"
                "• 'Расскажи о новостях в технологиях'\n"
                "• 'Что происходит в экономике?'\n"
                "• 'Какая статистика по источникам?'\n\n"
                "Я с удовольствием помогу!")
    else:
        return ("Спасибо за ваш вопрос! 😊\n\n"
                "К сожалению, сейчас в ленте нет новостей - они еще загружаются. "
                "Это может занять несколько минут при первом запуске.\n\n"
                "Как только новости загрузятся, я смогу помочь вам с:\n"
                "• Анализом новостей и трендов\n"
                "• Поиском по темам\n"
                "• Обсуждением текущих событий\n"
                "• Ответами на ваши вопросы\n\n"
                "Попробуйте обновить страницу через минуту-две, и я буду готов помочь!")


def _prepare_news_context(articles: list) -> str:
    """Подготавливает контекст из новостей для AI"""
    if not articles:
        return "В данный момент в ленте нет новостей."
    
    # Собираем информацию о новостях
    topics = {}
    sources = {}
    recent_articles = []
    
    for article in articles[:30]:
        # Темы
        for topic in article.topics:
            topics[topic] = topics.get(topic, 0) + 1
        
        # Источники
        source_name = article.source_name or "Неизвестно"
        sources[source_name] = sources.get(source_name, 0) + 1
        
        # Свежие новости
        if article.published_at:
            try:
                if isinstance(article.published_at, str):
                    pub_date = datetime.fromisoformat(article.published_at.replace('Z', '+00:00')).replace(tzinfo=None)
                else:
                    pub_date = article.published_at
                if (datetime.utcnow() - pub_date).days < 1:
                    recent_articles.append({
                        "title": article.title,
                        "summary": article.summary or "",
                        "topics": article.topics,
                        "source": source_name
                    })
            except (ValueError, AttributeError):
                pass
    
    context = f"Контекст новостной ленты:\n"
    context += f"- Всего статей: {len(articles)}\n"
    context += f"- Свежих новостей (за 24 часа): {len(recent_articles)}\n"
    
    if topics:
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]
        context += f"- Основные темы: {', '.join([f'{t} ({c})' for t, c in top_topics])}\n"
    
    if sources:
        top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]
        context += f"- Основные источники: {', '.join([f'{s} ({c})' for s, c in top_sources])}\n"
    
    if recent_articles:
        context += f"\nПримеры свежих новостей:\n"
        for i, art in enumerate(recent_articles[:5], 1):
            context += f"{i}. {art['title']}\n"
            if art['summary']:
                context += f"   {art['summary'][:150]}...\n"
            context += f"   Темы: {', '.join(art['topics'][:3])}\n"
            context += f"   Источник: {art['source']}\n\n"
    
    return context


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Эндпоинт для общения с AI ассистентом"""
    # Всегда используем локальную логику для скорости и надежности
    articles = cache.get_articles()
    response_message = _generate_ai_response(request.message, articles)
    return ChatResponse(message=response_message)
