"""
Azure OpenAI embedding provider with endpoint switching support.
"""

from typing import List, Optional
from src.embeddings.base import BaseEmbeddingProvider
from src.core.exceptions import EmbeddingError, ConfigurationError

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


class AzureEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using Azure OpenAI service."""
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-08-01-preview",
        deployment_name: str = "text-embedding-ada-002",
        dimension: int = 1536
    ):
        if AzureOpenAI is None:
            raise EmbeddingError("openai SDK is required for AzureEmbeddingProvider.")
        
        self._endpoint = endpoint
        self._api_key = api_key
        self._api_version = api_version
        self._deployment_name = deployment_name
        self._dimension = dimension

        if not self._endpoint or not self._api_key:
            raise ConfigurationError("Azure OpenAI endpoint and api_key must be configured for embeddings.")

        try:
            self._client = AzureOpenAI(
                azure_endpoint=self._endpoint,
                api_key=self._api_key,
                api_version=self._api_version
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize AzureOpenAI client: {str(e)}")

    def update_endpoint(self, new_endpoint: str, new_api_key: Optional[str] = None, new_deployment: Optional[str] = None):
        """Dynamic endpoint switching at runtime."""
        self._endpoint = new_endpoint
        if new_api_key:
            self._api_key = new_api_key
        if new_deployment:
            self._deployment_name = new_deployment
            
        self._client = AzureOpenAI(
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version
        )

    def embed_text(self, text: str) -> List[float]:
        try:
            response = self._client.embeddings.create(
                input=[text],
                model=self._deployment_name
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"Azure embedding failed for text: {str(e)}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            # Azure OpenAI supports batch inputs up to 2048 items
            response = self._client.embeddings.create(
                input=texts,
                model=self._deployment_name
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingError(f"Azure batch embedding failed: {str(e)}")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return f"azure-{self._deployment_name}"
