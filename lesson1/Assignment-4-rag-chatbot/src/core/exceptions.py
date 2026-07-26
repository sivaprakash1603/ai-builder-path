"""
Core domain exceptions for the RAG chatbot system.
"""

class RAGException(Exception):
    """Base exception class for all RAG chatbot errors."""
    pass


class ConfigurationError(RAGException):
    """Raised when configuration or environment variables are invalid/missing."""
    pass


class IngestionError(RAGException):
    """Raised when document loading, parsing, or chunking fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails across any provider."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector database indexing, querying, or persistence fails."""
    pass


class LLMProviderError(RAGException):
    """Raised when communication with Claude, Azure OpenAI, or Ollama fails."""
    pass


class RetrievalError(RAGException):
    """Raised when document retrieval or similarity search fails."""
    pass
