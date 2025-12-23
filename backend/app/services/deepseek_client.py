"""Сервис для работы с DeepSeek API"""
import httpx
from typing import List, Dict, Any
from ..config import get_settings

settings = get_settings()


async def call_deepseek_api(
    messages: List[Dict[str, str]],
    model: str = "deepseek-chat"
) -> str:
    """
    Вызывает DeepSeek API для генерации ответа
    
    Args:
        messages: Список сообщений в формате [{"role": "user", "content": "..."}]
        model: Модель DeepSeek для использования (по умолчанию "deepseek-chat")
    
    Returns:
        Ответ от AI модели
    """
    if not settings.deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY не настроен. Установите переменную окружения DEEPSEEK_API_KEY.")
    
    api_url = f"{settings.deepseek_api_url}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.deepseek_api_key}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Неожиданный формат ответа от DeepSeek API: {data}")
                
        except httpx.HTTPStatusError as e:
            # Формируем понятное сообщение об ошибке
            error_msg = f"Ошибка HTTP от DeepSeek API: {e.response.status_code}"
            if e.response.text:
                try:
                    error_data = e.response.json()
                    if "error" in error_data:
                        error_detail = error_data['error'].get('message', 'Unknown error')
                        error_msg += f" - {error_detail}"
                        # Для ошибок баланса (402) или других проблем с API, выбрасываем специальное исключение
                        if e.response.status_code == 402:
                            error_msg = f"DeepSeek API: Недостаточно средств на балансе (402)"
                except:
                    error_msg += f" - {e.response.text}"
            # Выбрасываем ValueError, который будет обработан в main.py и переключит на fallback
            raise ValueError(error_msg)
        except httpx.RequestError as e:
            raise ValueError(f"Ошибка подключения к DeepSeek API: {str(e)}")

