"""
Azure OpenAI LLM provider with dynamic endpoint switching support.
Implements the reference design pattern for switching between multiple Azure OpenAI resources,
deployments (e.g. gpt-4o, gpt-35-turbo), and regions at runtime.
"""

from typing import Dict, Any, Optional, List
from src.llm.base import BaseLLMProvider
from src.core.types import ChatMessage
from src.core.exceptions import LLMProviderError, ConfigurationError

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


class AzureOpenAILLMProvider(BaseLLMProvider):
    """
    Azure OpenAI chat completion provider supporting dynamic endpoint switching.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-08-01-preview",
        deployment_name: str = "gpt-4o"
    ):
        if AzureOpenAI is None:
            raise LLMProviderError("openai package is required for AzureOpenAILLMProvider.")

        self._endpoint = endpoint
        self._api_key = api_key
        self._api_version = api_version
        self._deployment_name = deployment_name

        if not self._endpoint or not self._api_key:
            raise ConfigurationError("Azure OpenAI endpoint and api_key must be provided.")

        try:
            self._client = AzureOpenAI(
                azure_endpoint=self._endpoint,
                api_key=self._api_key,
                api_version=self._api_version
            )
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize AzureOpenAI client: {str(e)}")

    def update_endpoint(
        self,
        new_endpoint: str,
        new_api_key: Optional[str] = None,
        new_deployment_name: Optional[str] = None,
        new_api_version: Optional[str] = None
    ) -> None:
        """
        Switch Azure OpenAI endpoints dynamically at runtime.
        This allows seamless failover or multi-region routing without downtime.
        """
        self._endpoint = new_endpoint
        if new_api_key:
            self._api_key = new_api_key
        if new_deployment_name:
            self._deployment_name = new_deployment_name
        if new_api_version:
            self._api_version = new_api_version

        try:
            self._client = AzureOpenAI(
                azure_endpoint=self._endpoint,
                api_key=self._api_key,
                api_version=self._api_version
            )
        except Exception as e:
            raise LLMProviderError(f"Failed to switch Azure OpenAI endpoint to '{new_endpoint}': {str(e)}")

    def generate(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            formatted_messages.append({
                "role": msg.role.lower(),
                "content": msg.content
            })

        try:
            response = self._client.chat.completions.create(
                model=self._deployment_name,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            choice = response.choices[0]
            usage = response.usage
            
            return {
                "text": choice.message.content or "",
                "model": response.model or self._deployment_name,
                "provider": "azure_openai",
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0
                }
            }
        except Exception as e:
            raise LLMProviderError(f"Azure OpenAI generation failed on deployment '{self._deployment_name}': {str(e)}")

    @property
    def provider_name(self) -> str:
        return "azure_openai"

    @property
    def current_model(self) -> str:
        return self._deployment_name
