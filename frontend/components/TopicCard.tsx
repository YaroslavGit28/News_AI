"use client";

import { Article, submitFeedback } from "../lib/api";
import { useMemo, useState } from "react";
import { useEffect, useRef } from "react";

type Props = {
  article: Article;
  onHide?: (payload: { id: number; title: string }) => void;
  viewMode?: "grid" | "list";
  saved?: boolean;
  onToggleSave?: (id: number) => void;
};

function formatTimeAgo(dateStr: string | null): string {
  if (!dateStr) return "Дата неизвестна";
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return "только что";
  if (diffMins < 60) return `${diffMins} мин. назад`;
  if (diffHours < 24) return `${diffHours} ч. назад`;
  return `${diffDays} дн. назад`;
}

function getFreshnessClass(dateStr: string | null): string {
  if (!dateStr) return "freshness-unknown";
  const date = new Date(dateStr);
  const now = new Date();
  const diffHours = (now.getTime() - date.getTime()) / 3600000;

  if (diffHours < 1) return "freshness-very-fresh";
  if (diffHours < 6) return "freshness-fresh";
  if (diffHours < 24) return "freshness-recent";
  return "freshness-old";
}

function estimateReadingTime(text: string): string {
  const words = text.split(/\s+/).length;
  const minutes = Math.max(1, Math.round(words / 160));
  return `${minutes} мин. чтения`;
}

function getSentimentBadge(value: number | null): { label: string; className: string } {
  if (value === null || value === undefined) return { label: "нейтрально", className: "sentiment-neutral" };
  if (value > 0.2) return { label: "позитив", className: "sentiment-positive" };
  if (value < -0.2) return { label: "негатив", className: "sentiment-negative" };
  return { label: "нейтрально", className: "sentiment-neutral" };
}

const TOPIC_BACKGROUND: Record<string, string> = {
  технологии: "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1400&q=80",
  technology: "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1400&q=80",
  экономика: "https://images.unsplash.com/photo-1454165205744-3b78555e5572?auto=format&fit=crop&w=1400&q=80",
  economy: "https://images.unsplash.com/photo-1454165205744-3b78555e5572?auto=format&fit=crop&w=1400&q=80",
  культура: "https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&w=1400&q=80",
  culture: "https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&w=1400&q=80",
  спорт: "https://images.unsplash.com/photo-1517649763962-0c623066013b?auto=format&fit=crop&w=1400&q=80",
  sport: "https://images.unsplash.com/photo-1517649763962-0c623066013b?auto=format&fit=crop&w=1400&q=80",
  политика: "https://images.unsplash.com/photo-1469474968028-56623f02e42e?auto=format&fit=crop&w=1400&q=80",
  politics: "https://images.unsplash.com/photo-1469474968028-56623f02e42e?auto=format&fit=crop&w=1400&q=80",
  наука: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1400&q=80",
  science: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1400&q=80",
  бизнес: "https://images.unsplash.com/photo-1434030216411-0b793f4b4173?auto=format&fit=crop&w=1400&q=80",
  business: "https://images.unsplash.com/photo-1434030216411-0b793f4b4173?auto=format&fit=crop&w=1400&q=80",
  медиа: "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1400&q=80",
  media: "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1400&q=80"
};

function getBackgroundImage(article: Article): string {
  if (article.image_url) return article.image_url;
  const topicMatch = article.topics.find((topic) => TOPIC_BACKGROUND[topic.toLowerCase()]);
  if (topicMatch) {
    return TOPIC_BACKGROUND[topicMatch.toLowerCase()];
  }
  return `https://source.unsplash.com/featured/900x600?${encodeURIComponent(article.topics[0] ?? "news")}`;
}

function stripHtml(value: string): string {
  return value.replace(/<\/?[^>]+(>|$)/g, "");
}

export function TopicCard({ article, onHide, viewMode = "grid", saved = false, onToggleSave }: Props) {
  const [feedbackSent, setFeedbackSent] = useState<string | null>(null);
  const [isHiding, setIsHiding] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const imgRef = useRef<HTMLDivElement>(null);
  const previewImage = useMemo(() => getBackgroundImage(article), [article]);

  // Lazy loading изображений с Intersection Observer
  useEffect(() => {
    if (!imgRef.current || imageLoaded) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setImageLoaded(true);
            observer.disconnect();
          }
        });
      },
      { rootMargin: '50px' }
    );

    observer.observe(imgRef.current);
    return () => observer.disconnect();
  }, [imageLoaded]);
  const summaryText = useMemo(() => {
    const cleaned = stripHtml(article.summary ?? "");
    return cleaned || "Описание скоро появится.";
  }, [article.summary]);
  const timeAgo = formatTimeAgo(article.published_at);
  const freshnessClass = getFreshnessClass(article.published_at);
  const sentiment = getSentimentBadge(article.sentiment);
  const readingTime = estimateReadingTime(summaryText);

  const handleFeedback = async (action: "like" | "dislike" | "hide") => {
    try {
      await submitFeedback(article.id, action);
      setFeedbackSent(action);
      if (action === "hide" && onHide) {
        setIsHiding(true);
        setTimeout(
          () =>
            onHide({
              id: article.id,
              title: article.title
            }),
          300
        );
      }
    } catch (error) {
      console.error("Ошибка отправки обратной связи:", error);
    }
  };

  if (isHiding) return null;

  return (
    <article className={`topic-card ${viewMode === "list" ? "list" : ""} ${saved ? "saved" : ""}`}>
      <div 
        ref={imgRef}
        className="card-media" 
        style={{ 
          backgroundImage: imageLoaded ? `url("${previewImage}")` : undefined,
          backgroundColor: imageLoaded ? undefined : '#e2e8f0'
        }}
      >
        <div className="media-overlay">
          <div className="media-top">
            <span className="source-pill">{article.source_name || "Источник"}</span>
            <span className={`freshness ${freshnessClass}`} title={article.published_at || undefined}>
              {timeAgo}
            </span>
          </div>
          <a className="media-title" href={article.url} target="_blank" rel="noreferrer">
            {article.title}
          </a>
          <div className="topic-tags">
            {article.topics.slice(0, 3).map((topic) => (
              <span key={topic} className="topic-badge">
                {topic}
              </span>
            ))}
          </div>
        </div>
      </div>
      <div className="card-body">
        <div className="card-meta">
          <span>{article.source_name || "Источник"}</span>
          <span className={`sentiment ${sentiment.className}`}>{sentiment.label}</span>
          <span>{readingTime}</span>
        </div>
        <p>{summaryText}</p>
      </div>
      <footer>
        <div className="entities">
          <strong>Сущности:</strong> {article.entities.join(", ") || "нет"}
        </div>
        <div className="feedback-actions">
          {onToggleSave && (
            <button
              className={`feedback-btn bookmark ${saved ? "active" : ""}`}
              onClick={() => onToggleSave(article.id)}
              title={saved ? "Удалить из избранного" : "Сохранить"}
              aria-pressed={saved}
            >
              {saved ? "★" : "☆"}
            </button>
          )}
          <button
            className={`feedback-btn like ${feedbackSent === "like" ? "active" : ""}`}
            onClick={() => handleFeedback("like")}
            title="Нравится"
          >
            👍
          </button>
          <button
            className={`feedback-btn dislike ${feedbackSent === "dislike" ? "active" : ""}`}
            onClick={() => handleFeedback("dislike")}
            title="Не нравится"
          >
            👎
          </button>
          <button className="feedback-btn hide" onClick={() => handleFeedback("hide")} title="Скрыть">
            ✕
          </button>
        </div>
      </footer>
    </article>
  );
}
