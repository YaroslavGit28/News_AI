"use client";

type SortOption = "home" | "fresh" | "source";
type ViewMode = "grid" | "list";

type Props = {
  sortBy: SortOption;
  onSortChange: (value: SortOption) => void;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  visibleCount: number;
  totalCount: number;
  savedCount: number;
};

const sortLabels: Record<SortOption, string> = {
  home: "Главная",
  fresh: "Сначала свежие",
 
};

export function FeedToolbar({
  sortBy,
  onSortChange,
  viewMode,
  onViewModeChange,
  visibleCount,
  totalCount,
  savedCount
}: Props) {
  return (
    <section className="feed-toolbar" aria-label="Панель управления лентой">
      <div className="toolbar-col stats">
        <span className="toolbar-metric">
          Показано <strong>{visibleCount}</strong> / {totalCount}
        </span>
        {savedCount > 0 && (
          <span className="saved-chip" title="Сохранено в быстрый доступ">
            ★ {savedCount} избранных
          </span>
        )}
      </div>
      <div className="toolbar-col controls">
        <div className="sort-switch">
          {Object.entries(sortLabels).map(([value, label]) => (
            <button
              key={value}
              className={`pill-button ${sortBy === value ? "active" : ""}`}
              onClick={() => onSortChange(value as SortOption)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="view-toggle" role="group" aria-label="Отображение">
          <button
            className={`toggle-btn ${viewMode === "grid" ? "active" : ""}`}
            onClick={() => onViewModeChange("grid")}
            aria-pressed={viewMode === "grid"}
          >
            ▦
          </button>
          <button
            className={`toggle-btn ${viewMode === "list" ? "active" : ""}`}
            onClick={() => onViewModeChange("list")}
            aria-pressed={viewMode === "list"}
          >
            ☰
          </button>
        </div>
      </div>
    </section>
  );
}



