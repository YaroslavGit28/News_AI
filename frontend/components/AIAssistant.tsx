"use client";

import { useState, useRef, useEffect } from "react";
import { sendChatMessage, ChatMessage as APIChatMessage } from "../lib/api";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
};

export function AIAssistant() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "1",
      role: "assistant",
      content: "Привет! Я ваш AI-ассистент. Я могу помочь вам обсудить новости, проанализировать тренды, поговорить о будущем и многое другое. О чем вы хотели бы поговорить?",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
      inputRef.current?.focus();
    }
  }, [messages, isOpen]);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      // Подготавливаем историю для отправки (без id и timestamp)
      const history: APIChatMessage[] = messages
        .slice(1) // Пропускаем первое приветственное сообщение
        .map(msg => ({
          role: msg.role,
          content: msg.content
        }));
      
      const response = await sendChatMessage(userMessage.content, history);
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.message,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Ошибка отправки сообщения:", error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Извините, произошла ошибка при обработке вашего запроса. Попробуйте еще раз.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClearChat = () => {
    setMessages([
      {
        id: "1",
        role: "assistant",
        content: "Привет! Я ваш AI-ассистент. Я могу помочь вам обсудить новости, проанализировать тренды, поговорить о будущем и многое другое. О чем вы хотели бы поговорить?",
        timestamp: new Date(),
      },
    ]);
  };

  return (
    <>
      <button
        className="ai-assistant-button"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Открыть AI ассистент"
      >
        <span className="ai-button-icon">🤖</span>
        <span className="ai-button-text">AI Ассистент</span>
      </button>

      {isOpen && (
        <div className="ai-assistant-panel">
          <div className="ai-assistant-header">
            <div className="ai-assistant-title">
              <span className="ai-header-icon">🤖</span>
              <div>
                <h3>AI Ассистент</h3>
                <p>Обсуждение новостей и аналитика</p>
              </div>
            </div>
            <div className="ai-assistant-actions">
              <button
                className="ai-clear-btn"
                onClick={handleClearChat}
                title="Очистить чат"
              >
                🗑️
              </button>
              <button
                className="ai-close-btn"
                onClick={() => setIsOpen(false)}
                aria-label="Закрыть"
              >
                ✕
              </button>
            </div>
          </div>

          <div className="ai-assistant-messages">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`ai-message ai-message-${message.role}`}
              >
                <div className="ai-message-content">
                  {message.content}
                </div>
                <div className="ai-message-time">
                  {message.timestamp.toLocaleTimeString("ru-RU", {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="ai-message ai-message-assistant">
                <div className="ai-message-content ai-loading">
                  <span className="ai-typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="ai-assistant-input-container">
            <input
              ref={inputRef}
              type="text"
              className="ai-assistant-input"
              placeholder="Напишите сообщение..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
            />
            <button
              className="ai-send-btn"
              onClick={handleSend}
              disabled={isLoading || !inputValue.trim()}
              aria-label="Отправить сообщение"
            >
              ➤
            </button>
          </div>
        </div>
      )}
    </>
  );
}

