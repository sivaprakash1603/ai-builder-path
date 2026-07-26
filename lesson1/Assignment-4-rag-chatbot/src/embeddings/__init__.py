from typing import Optional
from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.local_embedding import LocalEmbeddingProvider
from src.embeddings.azure_embedding import AzureEmbeddingProvider
from src.embeddings.ollama_embedding import OllamaEmbeddingProvider
from src.config.settings import settings


class EmbeddingProviderFactory:
    """Factory class to instantiate embedding providers dynamically."""
    
    @staticmethod
    def get_provider(provider_name: Optional[str] = None) -> BaseEmbeddingProvider:
        provider = (provider_name or settings.default_embedding_provider).lower()
        
        if provider == "local":
            return LocalEmbeddingProvider(model_name=settings.local_embedding_model)
        elif provider in ("azure_openai", "azure"):
            return AzureEmbeddingProvider(
                endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                deployment_name=settings.azure_openai_embedding_deployment
            )
        elif provider == "ollama":
            return OllamaEmbeddingProvider(
                base_url=settings.ollama_base_url,
                model_name=settings.ollama_embedding_model
            )
        else:
            # Default fallback to Local Embedding Provider
            return LocalEmbeddingProvider(model_name=settings.local_embedding_model)


__all__ = [
    "BaseEmbeddingProvider",
    "LocalEmbeddingProvider",
    "AzureEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "EmbeddingProviderFactory"
]
