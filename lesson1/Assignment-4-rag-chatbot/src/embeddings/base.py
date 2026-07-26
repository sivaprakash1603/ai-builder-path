"""
Abstract base interface for vector embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract interface for text embedding generation."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate a vector embedding for a single string of text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of strings."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the generated vectors."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the embedding provider."""
        pass
