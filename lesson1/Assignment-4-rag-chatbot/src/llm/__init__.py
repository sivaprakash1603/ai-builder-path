from typing import Optional
from src.llm.base import BaseLLMProvider
from src.llm.anthropic_client import AnthropicLLMProvider
from src.llm.azure_client import AzureOpenAILLMProvider
from src.llm.ollama_client import OllamaLLMProvider
from src.config.settings import settings
from src.core.exceptions import ConfigurationError


class LLMProviderFactory:
    """Factory class to instantiate LLM chat providers dynamically at runtime."""

    @staticmethod
    def get_provider(provider_name: Optional[str] = None) -> BaseLLMProvider:
        provider = (provider_name or settings.default_llm_provider).lower()

        if provider == "anthropic":
            return AnthropicLLMProvider(
                base_url=settings.anthropic_base_url,
                auth_token=settings.anthropic_auth_token or settings.anthropic_api_key,
                model_name=settings.anthropic_model
            )
        elif provider in ("azure_openai", "azure"):
            return AzureOpenAILLMProvider(
                endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                deployment_name=settings.azure_openai_deployment_name
            )
        elif provider == "ollama":
            return OllamaLLMProvider(
                base_url=settings.ollama_base_url,
                model_name=settings.ollama_model
            )
        else:
            # Fallback to Anthropic Claude
            return AnthropicLLMProvider(
                base_url=settings.anthropic_base_url,
                auth_token=settings.anthropic_auth_token or settings.anthropic_api_key,
                model_name=settings.anthropic_model
            )


__all__ = [
    "BaseLLMProvider",
    "AnthropicLLMProvider",
    "AzureOpenAILLMProvider",
    "OllamaLLMProvider",
    "LLMProviderFactory"
]
