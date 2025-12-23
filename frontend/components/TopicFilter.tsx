"use client";

type TopicInfo = {
  name: string;
  count: number;
};

type Props = {
  topics: TopicInfo[];
  selectedTopics: Set<string>;
  onToggleTopic: (topic: string) => void;
  onClearAll: () => void;
};

const ALL_TOPICS = [
  "Общее",
  "Технологии",
  "Кибербезопасность",
  "Экономика",
  "Общество",
  "Культура",
  "Наука",
  "Спорт",
];

export function TopicFilter({ topics, selectedTopics, onToggleTopic, onClearAll }: Props) {
  const topicCountMap = new Map(topics.map((t) => [t.name, t.count]));
  
  const allTopicsWithCounts = ALL_TOPICS.map((name) => ({
    name,
    count: topicCountMap.get(name) ?? 0
  }));

  return (
    <div className="topic-filter">
      <div className="filter-header">
        <h3>Темы</h3>
        {selectedTopics.size > 0 && (
          <button className="clear-filter" onClick={onClearAll}>
            Сбросить
          </button>
        )}
      </div>
      <div className="filter-tags">
        {allTopicsWithCounts.map((topic) => (
          <button
            key={topic.name}
            className={`filter-tag ${selectedTopics.has(topic.name) ? "active" : ""}`}
            onClick={() => onToggleTopic(topic.name)}
          >
            {topic.name}
          </button>
        ))}
      </div>
    </div>
  );
}


