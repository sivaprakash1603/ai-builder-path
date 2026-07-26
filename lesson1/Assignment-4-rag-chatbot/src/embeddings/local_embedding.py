"""
Local open-source embedding provider using SentenceTransformers.
Runs entirely offline without API keys or rate limits.
"""

from typing import List
from src.embeddings.base import BaseEmbeddingProvider
from src.core.exceptions import EmbeddingError

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding generator using HuggingFace SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise EmbeddingError("sentence-transformers package is required for LocalEmbeddingProvider.")
        self._model_name = model_name
        try:
            # Load model (cached locally after first download)
            self._model = SentenceTransformer(self._model_name)
            self._dimension = getattr(self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension)()
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize local embedding model '{model_name}': {str(e)}")

    def embed_text(self, text: str) -> List[float]:
        try:
            embedding = self._model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Local embedding generation failed for text: {str(e)}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self._model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            raise EmbeddingError(f"Local batch embedding generation failed: {str(e)}")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return f"local-{self._model_name}"
