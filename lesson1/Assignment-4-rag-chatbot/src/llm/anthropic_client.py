"""
Anthropic (Claude) LLM client with support for custom base URL proxies and authentication tokens.
Designed to handle enterprise proxy gateways (e.g. Presidio LLM gateway) seamlessly.
"""

import os
import httpx
from typing import Dict, Any, Optional, List
from src.llm.base import BaseLLMProvider
from src.core.types import ChatMessage
from src.core.exceptions import LLMProviderError, ConfigurationError

try:
    import anthropic
except ImportError:
    anthropic = None


class AnthropicLLMProvider(BaseLLMProvider):
    """
    Claude provider supporting ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        model_name: str = "claude-3-5-sonnet-20241022"
    ):
        self._base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        self._api_key = auth_token or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
        self._model_name = model_name

        if not self._api_key:
            raise ConfigurationError(
                "Anthropic auth token/API key missing. Please set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY."
            )

        # Initialize Anthropic client with custom base URL and headers if SDK available
        if anthropic:
            try:
                # Custom headers ensure compatibility whether proxy requires x-api-key or Authorization Bearer
                headers = {
                    "x-api-key": self._api_key,
                    "Authorization": f"Bearer {self._api_key}"
                }
                self._client = anthropic.Anthropic(
                    base_url=self._base_url,
                    api_key=self._api_key,
                    default_headers=headers
                )
            except Exception as e:
                self._client = None
        else:
            self._client = None

    def update_proxy(self, base_url: str, auth_token: str, model_name: Optional[str] = None):
        """Dynamic runtime switching of proxy endpoint and authentication token."""
        self._base_url = base_url
        self._api_key = auth_token
        if model_name:
            self._model_name = model_name
            
        if anthropic:
            headers = {
                "x-api-key": self._api_key,
                "Authorization": f"Bearer {self._api_key}"
            }
            self._client = anthropic.Anthropic(
                base_url=self._base_url,
                api_key=self._api_key,
                default_headers=headers
            )

    def _generate_via_httpx(
        self,
        formatted_messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Direct HTTP fallback for proxy endpoints with custom auth structures."""
        url = f"{self._base_url.rstrip('/')}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "Authorization": f"Bearer {self._api_key}",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": self._model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": formatted_messages
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                
                content_text = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content_text += block.get("text", "")
                
                usage = data.get("usage", {})
                return {
                    "text": content_text,
                    "model": data.get("model", self._model_name),
                    "provider": "anthropic-proxy",
                    "usage": {
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0)
                    }
                }
        except Exception as e:
            raise LLMProviderError(f"HTTP fallback to Anthropic proxy failed: {str(e)}")

    def generate(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        formatted_messages = []
        for msg in messages:
            # Skip system messages in messages array for Anthropic v1/messages API
            if msg.role.lower() == "system":
                if not system_prompt:
                    system_prompt = msg.content
                continue
            formatted_messages.append({
                "role": msg.role.lower() if msg.role.lower() in ("user", "assistant") else "user",
                "content": msg.content
            })

        if not formatted_messages:
            raise LLMProviderError("No valid user/assistant messages provided for Anthropic generation.")

        # Attempt generation via official SDK first
        if self._client:
            try:
                kwargs = {
                    "model": self._model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": formatted_messages
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                resp = self._client.messages.create(**kwargs)
                content_text = ""
                for block in resp.content:
                    if getattr(block, "type", None) == "text":
                        content_text += block.text

                return {
                    "text": content_text,
                    "model": resp.model,
                    "provider": "anthropic",
                    "usage": {
                        "prompt_tokens": getattr(resp.usage, "input_tokens", 0),
                        "completion_tokens": getattr(resp.usage, "output_tokens", 0)
                    }
                }
            except Exception as e:
                # If SDK fails due to proxy header mismatch or path routing, fallback to HTTP
                try:
                    return self._generate_via_httpx(formatted_messages, system_prompt, temperature, max_tokens)
                except Exception as http_err:
                    raise LLMProviderError(f"Anthropic generation failed (SDK: {str(e)} | HTTP: {str(http_err)})")
        else:
            return self._generate_via_httpx(formatted_messages, system_prompt, temperature, max_tokens)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def current_model(self) -> str:
        return self._model_name
