"use client";

import { HiddenRecord } from "../lib/api";

type Props = {
  items: HiddenRecord[];
  resolveTitle: (id: number) => string;
  onRestore: (id: number) => void;
  currentTick: number;
};

function formatCountdown(expiresAt: string, now: number): string {
  const diff = new Date(expiresAt).getTime() - now;
  if (diff <= 0) return "время истекло";
  const minutes = Math.floor(diff / 60000);
  const seconds = Math.floor((diff % 60000) / 1000)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${seconds}`;
}

export function HiddenPanel({ items, onRestore, resolveTitle, currentTick }: Props) {
  if (!items.length) return null;

  return (
    <div className="hidden-panel">
      <h4>Скрытые материалы (можно вернуть в течение часа)</h4>
      <ul>
        {items.map((item) => (
          <li key={item.article_id}>
            <div>
              <span className="hidden-title">{resolveTitle(item.article_id)}</span>
              <span className="hidden-timer">{formatCountdown(item.expires_at, currentTick)}</span>
            </div>
            <button onClick={() => onRestore(item.article_id)}>Вернуть</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

