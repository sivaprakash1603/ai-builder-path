"""
Abstract base interface for Vector Store database implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.core.types import TextChunk, SearchResult


class BaseVectorStore(ABC):
    """Abstract interface for storing text chunks and vector embeddings."""
    
    @abstractmethod
    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """Add text chunks with embeddings to the vector store. Returns count added."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        similarity_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search top-K most similar chunks using cosine similarity."""
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks associated with a specific doc_id. Returns count deleted."""
        pass

    @abstractmethod
    def list_documents(self) -> List[Dict[str, Any]]:
        """Return summary list of indexed documents (doc_id, title, source, chunk_count)."""
        pass

    @abstractmethod
    def get_chunk_count(self) -> int:
        """Return total number of chunks currently stored."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all chunks and reset the vector store."""
        pass
