"""
Ollama local LLM provider using REST API.
Enables offline privacy-first chat completions.
"""

import httpx
from typing import Dict, Any, Optional, List
from src.llm.base import BaseLLMProvider
from src.core.types import ChatMessage
from src.core.exceptions import LLMProviderError


class OllamaLLMProvider(BaseLLMProvider):
    """
    Ollama local LLM chat completion provider.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.1:8b"):
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name

    def update_model(self, new_model_name: str, new_base_url: Optional[str] = None):
        """Switch Ollama models or server endpoints dynamically."""
        self._model_name = new_model_name
        if new_base_url:
            self._base_url = new_base_url.rstrip("/")

    def generate(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        url = f"{self._base_url}/api/chat"
        
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
            
        for msg in messages:
            formatted_messages.append({
                "role": msg.role.lower(),
                "content": msg.content
            })

        payload = {
            "model": self._model_name,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                
                message_content = data.get("message", {}).get("content", "")
                
                return {
                    "text": message_content,
                    "model": data.get("model", self._model_name),
                    "provider": "ollama",
                    "usage": {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0)
                    }
                }
        except Exception as e:
            raise LLMProviderError(f"Ollama chat completion failed for model '{self._model_name}': {str(e)}")

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def current_model(self) -> str:
        return self._model_name
