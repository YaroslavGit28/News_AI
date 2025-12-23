"use client";

type Props = {
  onRefresh?: () => void;
  refreshing?: boolean;
  searchQuery: string;
  onSearchChange: (value: string) => void;
  articleCount: number;
  lastUpdated: Date | null;
};

export function AppHeader({ onRefresh, refreshing, searchQuery, onSearchChange, articleCount, lastUpdated }: Props) {
  const formattedUpdated = lastUpdated
    ? new Intl.DateTimeFormat("ru-RU", {
        hour: "2-digit",
        minute: "2-digit"
      }).format(lastUpdated)
    : "—";

  return (
    <header className="app-header">
      <div className="logo-block">
        <div className="logo-dot" />
        <div>
          <span className="brand">Persona News</span>
          <p>Персональный ИИ-агрегатор</p>
        </div>
      </div>
      <div className="header-actions">
        <div className="header-meta">
          <span>В ленте {articleCount} материалов</span>
          <span className="dot" />
          <span>обновлено в {formattedUpdated}</span>
        </div>
        <input
          className="search-input"
          placeholder="Поиск по заголовку или источнику"
          type="search"
          value={searchQuery}
          onChange={(evt) => onSearchChange(evt.target.value)}
        />
        <div className="header-buttons">
          <button className="refresh-btn" onClick={onRefresh} disabled={refreshing}>
            ↻ {refreshing ? "Обновляем..." : "Обновить ленту"}
          </button>
        </div>
      </div>
    </header>
  );
}

