"use client";

import { useEffect, useState, useMemo } from "react";
import { fetchFeed, Article, HiddenRecord, submitFeedback } from "../lib/api";
import { TopicCard } from "../components/TopicCard";
import { TopicFilter } from "../components/TopicFilter";
import { AppHeader } from "../components/AppHeader";
import { HiddenPanel } from "../components/HiddenPanel";
import { FeedToolbar } from "../components/FeedToolbar";
import dynamic from "next/dynamic";

// Lazy load AI Assistant - загружается только при необходимости
const AIAssistant = dynamic(() => import("../components/AIAssistant").then(mod => ({ default: mod.AIAssistant })), {
  ssr: false,
  loading: () => null
});

export default function Page() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTopics, setSelectedTopics] = useState<Set<string>>(new Set());
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [hiddenRecords, setHiddenRecords] = useState<Record<number, HiddenRecord>>({});
  const [timerTick, setTimerTick] = useState(Date.now());
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedSearchQuery, setDebouncedSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<"home" | "fresh" | "source">("home");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [savedIds, setSavedIds] = useState<Set<number>>(new Set());
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  useEffect(() => {
    // Загружаем данные сразу
    loadFeed();
    // Обновляем каждые 2 минуты
    const interval = setInterval(loadFeed, 120000);
    return () => clearInterval(interval);
  }, []);



  useEffect(() => {
    const interval = setInterval(() => {
      setTimerTick(Date.now());
      setHiddenRecords((prev) => {
        const now = Date.now();
        let mutated = false;
        const next: Record<number, HiddenRecord> = {};
        Object.entries(prev).forEach(([key, record]) => {
          if (new Date(record.expires_at).getTime() > now) {
            next[Number(key)] = record;
          } else {
            mutated = true;
          }
        });
        return mutated ? next : prev;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const stored = window.localStorage.getItem("persona-news:saved");
      if (stored) {
        setSavedIds(new Set(JSON.parse(stored)));
      }
    } catch {
      setSavedIds(new Set());
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem("persona-news:saved", JSON.stringify(Array.from(savedIds)));
  }, [savedIds]);

  // Debounce для поиска - уменьшает количество перерисовок
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchQuery(searchQuery);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const loadFeed = async () => {
    try {
      setLoading((prev) => (articles.length ? prev : true));
      setRefreshing(true);
      const data = await fetchFeed(100);
      setArticles(data.articles);
      const hiddenMap: Record<number, HiddenRecord> = {};
      data.hidden.forEach((item) => {
        hiddenMap[item.article_id] = item;
      });
      setHiddenRecords(hiddenMap);
      setLastUpdated(new Date());
      setErrorMessage(null);
    } catch (error) {
      console.error("Ошибка загрузки ленты:", error);
      const message = error instanceof Error 
        ? error.message 
        : "Не удалось загрузить ленту";
      setErrorMessage(message);
      
      // Если это ошибка подключения, не показываем пустую ленту
      if (message.includes("подключиться") || message.includes("недоступен")) {
        setArticles([]);
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const baseVisibleArticles = useMemo(() => {
    const now = Date.now();
    const query = debouncedSearchQuery.trim().toLowerCase();
    return articles.filter((article) => {
      const hiddenEntry = hiddenRecords[article.id];
      if (hiddenEntry && new Date(hiddenEntry.expires_at).getTime() > now) {
        return false;
      }
      if (!query) return true;
      const haystack = [
        article.title,
        article.summary,
        article.source_name,
        article.entities.join(" "),
        article.topics.join(" ")
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });
      }, [articles, hiddenRecords, debouncedSearchQuery]);

  const topicBuckets = useMemo(() => {
    const buckets: Record<string, Article[]> = {};
    baseVisibleArticles.forEach((article) => {
      article.topics.forEach((topic) => {
        if (!buckets[topic]) buckets[topic] = [];
        buckets[topic].push(article);
      });
    });
    Object.values(buckets).forEach((bucket) =>
      bucket.sort((a, b) => {
        const dateA = new Date(a.published_at ?? a.created_at ?? "").getTime();
        const dateB = new Date(b.published_at ?? b.created_at ?? "").getTime();
        return dateB - dateA;
      })
    );
    return buckets;
  }, [baseVisibleArticles]);

  const topicStats = useMemo(() => {
    return Object.entries(topicBuckets)
      .map(([name, list]) => ({ name, count: list.length }))
      .sort((a, b) => b.count - a.count);
  }, [topicBuckets]);

  const filteredArticles = useMemo(() => {
    return baseVisibleArticles.filter((article) => {
      if (!selectedTopics.size) return true;
      return article.topics.some((topic) => selectedTopics.has(topic));
    });
  }, [baseVisibleArticles, selectedTopics]);

  const sortedArticles = useMemo(() => {
    const clone = [...filteredArticles];

    switch (sortBy) {
      case "fresh":
        return clone.sort((a, b) => {
          const dateA = new Date(a.published_at ?? a.created_at ?? "").getTime();
          const dateB = new Date(b.published_at ?? b.created_at ?? "").getTime();
          return dateB - dateA;
        });
      case "source":
        return clone.sort((a, b) => (a.source_name ?? "").localeCompare(b.source_name ?? ""));
      case "home":
      default:
        return clone.sort((a, b) => a.title.localeCompare(b.title, "ru"));
    }
  }, [filteredArticles, sortBy]);

  const handleToggleTopic = (topic: string) => {
    setSelectedTopics((prev) => {
      const next = new Set(prev);
      if (next.has(topic)) {
        next.delete(topic);
      } else {
        next.add(topic);
      }
      return next;
    });
  };

  const handleClearFilter = () => {
    setSelectedTopics(new Set());
  };

  const handleToggleSave = (id: number) => {
    setSavedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleHideArticle = ({ id }: { id: number; title: string }) => {
    setHiddenRecords((prev) => ({
      ...prev,
      [id]: {
        article_id: id,
        hidden_at: new Date().toISOString(),
        expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString()
      }
    }));
  };

  const handleRestoreArticle = async (id: number) => {
    try {
      await submitFeedback(id, "undo_hide");
      setHiddenRecords((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    } catch (error) {
      console.error("Не удалось восстановить статью", error);
    }
  };

  const resolveTitle = (id: number) => articles.find((a) => a.id === id)?.title ?? "Новость";

  const uniqueSortedArticles = useMemo(() => {
    const seen = new Set<number>();
    return sortedArticles.filter((article) => {
      if (seen.has(article.id)) return false;
      seen.add(article.id);
      return true;
    });
  }, [sortedArticles]);

  const limit = 50;

  const displayedArticles = uniqueSortedArticles.slice(0, limit);
  const savedArticles = articles.filter((article) => savedIds.has(article.id)).slice(0, 6);

  return (
    <div className="dashboard">
      <AppHeader
        onRefresh={loadFeed}
        refreshing={refreshing}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        articleCount={articles.length}
        lastUpdated={lastUpdated}
      />
      <main className="page">
        {errorMessage && (
          <div className="notice error">
            <div style={{ flex: 1 }}>
              <strong>Ошибка подключения</strong>
              <p style={{ margin: "4px 0 0", fontSize: "14px", opacity: 0.9 }}>
                {errorMessage}
              </p>
              {errorMessage.includes("недоступен") || errorMessage.includes("подключиться") ? (
                <p style={{ margin: "8px 0 0", fontSize: "12px", opacity: 0.8 }}>
                  💡 Убедитесь, что бэкенд запущен: <code>cd backend && uvicorn app.main:app --reload</code>
                </p>
              ) : null}
            </div>
            <button onClick={loadFeed}>🔄 Повторить</button>
          </div>
        )}
        <section className="hero">
          <div>
            <h1>Persona News</h1>
            <p>
              Персональная лента с более чем 15 мировыми и региональными источниками. Лайкните то, что интересно, —
              рекомендации адаптируются мгновенно.
            </p>
            <ul className="hero-points">
              <li>🧠 ИИ-сводки по ключевым темам и персональным интересам</li>
              <li>⚡ Автообновление каждые 2 минуты + цвет свежести</li>
              <li>⭐ Избранное и скрытые материалы в одном клике</li>
            </ul>
          </div>
          <div className="hero-stats">
            <div>
              <span className="stat-label">Всего статей</span>
              <strong>{articles.length}</strong>
            </div>
            <div>
              <span className="stat-label">Тем</span>
              <strong>{topicStats.length}</strong>
            </div>
            <div>
              <span className="stat-label">Обновление</span>
              <strong>{lastUpdated ? "только что" : "каждые 2 мин"}</strong>
            </div>
          </div>
        </section>

        {!!savedArticles.length && (
          <section className="saved-ribbon" aria-label="Сохраненные статьи">
            <header>
              <div>
                <h3>Избранное</h3>
                <p>Подборка для второго чтения</p>
              </div>
              <span>{savedArticles.length} материалов</span>
            </header>
            <div className="saved-scroll">
              {savedArticles.map((article) => (
                <article key={article.id} className="saved-card">
                  <span className="saved-source">{article.source_name ?? "Источник"}</span>
                  <a href={article.url} target="_blank" rel="noreferrer">
                    {article.title}
                  </a>
                  <button onClick={() => handleToggleSave(article.id)}>Убрать</button>
                </article>
              ))}
            </div>
          </section>
        )}

        {topicStats.length > 0 && (
          <TopicFilter
            topics={topicStats}
            selectedTopics={selectedTopics}
            onToggleTopic={handleToggleTopic}
            onClearAll={handleClearFilter}
          />
        )}

        <FeedToolbar
          sortBy={sortBy}
          onSortChange={setSortBy}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          visibleCount={displayedArticles.length}
          totalCount={filteredArticles.length}
          savedCount={savedIds.size}
        />

        {loading && articles.length === 0 ? (
          <div className="notice">
            <p>⏳ Загрузка новостей из источников...</p>
            <p style={{ marginTop: "8px", fontSize: "14px", opacity: 0.8 }}>
              Это может занять несколько минут при первом запуске. Данные загружаются в фоне.
            </p>
          </div>
        ) : filteredArticles.length === 0 ? (
          <p className="notice">
            {articles.length === 0
              ? "📰 Новости загружаются... Если это сообщение не исчезает, проверьте, что backend запущен и имеет доступ к интернету."
              : selectedTopics.size > 0
              ? "Нет статей по выбранным темам."
              : "Все статьи скрыты — верните их через панель снизу."}
          </p>
        ) : (
          <>
            <div className="feed-stats">
              Показано {displayedArticles.length} из {filteredArticles.length} статей
              {selectedTopics.size > 0 && ` • фильтр: ${Array.from(selectedTopics).join(", ")}`}
              {debouncedSearchQuery && ` • поиск: "${debouncedSearchQuery}"`}
            </div>
            <section className={`feed-grid ${viewMode === "list" ? "list-view" : ""}`}>
              {displayedArticles.map((article) => (
                <TopicCard
                  key={article.id}
                  article={article}
                  onHide={handleHideArticle}
                  viewMode={viewMode}
                  saved={savedIds.has(article.id)}
                  onToggleSave={handleToggleSave}
                />
              ))}
            </section>
          </>
        )}
        <HiddenPanel
          items={Object.values(hiddenRecords)}
          resolveTitle={resolveTitle}
          onRestore={handleRestoreArticle}
          currentTick={timerTick}
        />
      </main>
      <AIAssistant />
    </div>
  );
}

