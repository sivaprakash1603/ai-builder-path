"""
Document Retriever module for RAG pipeline.
Coordinates embedding generation of queries and similarity search in the vector store.
"""

from typing import List, Dict, Any, Optional
from src.embeddings.base import BaseEmbeddingProvider
from src.storage.base import BaseVectorStore
from src.core.types import SearchResult
from src.core.exceptions import RetrievalError


class DocumentRetriever:
    """
    Retriever that connects an embedding provider to a vector database.
    Performs vectorization of search queries and retrieves ranked semantic matches.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        default_top_k: int = 4,
        default_threshold: float = 0.25
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant document chunks for a given natural language query.
        """
        k = top_k if top_k is not None else self.default_top_k
        threshold = similarity_threshold if similarity_threshold is not None else self.default_threshold

        try:
            # 1. Embed the query string
            query_vector = self.embedding_provider.embed_text(query)
            
            # 2. Search vector store using cosine similarity
            results = self.vector_store.search(
                query_embedding=query_vector,
                top_k=k,
                similarity_threshold=threshold,
                filter_metadata=filter_metadata
            )
            return results
        except Exception as e:
            raise RetrievalError(f"Failed to retrieve documents for query '{query}': {str(e)}")

    def format_context(self, search_results: List[SearchResult]) -> str:
        """
        Format retrieved chunks into a clean, structured context string for LLM prompt injection.
        """
        if not search_results:
            return "No relevant context found in knowledge base."

        context_blocks = []
        for i, res in enumerate(search_results):
            title = res.metadata.get("title", res.doc_id)
            source = res.metadata.get("source", "Unknown source")
            block = (
                f"--- [Source {i+1}: {title} (Similarity: {res.score:.2f})] ---\n"
                f"{res.text.strip()}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)
