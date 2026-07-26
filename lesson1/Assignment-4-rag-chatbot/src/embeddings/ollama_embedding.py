"""
Ollama embedding provider using local HTTP REST API.
"""

import httpx
from typing import List
from src.embeddings.base import BaseEmbeddingProvider
from src.core.exceptions import EmbeddingError


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using local Ollama instance."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "nomic-embed-text", dimension: int = 768):
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        url = f"{self._base_url}/api/embeddings"
        payload = {
            "model": self._model_name,
            "prompt": text
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                if "embedding" not in data:
                    raise EmbeddingError("Ollama response missing 'embedding' field.")
                return data["embedding"]
        except Exception as e:
            raise EmbeddingError(f"Ollama embedding request failed: {str(e)}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Ollama /api/embeddings is sequential per text prompt in standard REST API
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return f"ollama-{self._model_name}"
