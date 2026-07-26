"""
End-to-End RAG Pipeline Orchestrator.
Coordinates document ingestion, chunking, embedding, vector storage, retrieval, and LLM chat response generation.
Built from scratch without LangChain or LlamaIndex.
"""

import time
from typing import List, Optional, Dict, Any
from src.core.types import Document, TextChunk, SearchResult, ChatMessage, ChatResponse
from src.core.exceptions import RAGException, IngestionError, RetrievalError, LLMProviderError
from src.ingestion.loaders import DocumentLoaderFactory
from src.ingestion.chunker import RecursiveCharacterChunker
from src.embeddings.base import BaseEmbeddingProvider
from src.storage.base import BaseVectorStore
from src.llm.base import BaseLLMProvider
from src.rag.retriever import DocumentRetriever
from src.rag.prompt_builder import RAGPromptBuilder
from src.config.settings import settings


class RAGPipeline:
    """
    Core RAG Engine orchestrating the complete lifecycle:
    Ingestion -> Chunking -> Embedding -> Storage -> Retrieval -> Generation.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        
        self.chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.retriever = DocumentRetriever(
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            default_top_k=settings.top_k_retrieval,
            default_threshold=settings.similarity_threshold
        )

    def switch_llm_provider(self, new_llm_provider: BaseLLMProvider) -> None:
        """Switch the active LLM provider (e.g. from Anthropic Claude to Azure OpenAI)."""
        self.llm_provider = new_llm_provider

    def switch_embedding_provider(self, new_embedding_provider: BaseEmbeddingProvider) -> None:
        """Switch embedding provider and update retriever binding."""
        self.embedding_provider = new_embedding_provider
        self.retriever.embedding_provider = new_embedding_provider

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load a file, chunk it, generate vector embeddings, and store in vector database.
        Returns summary dictionary.
        """
        start_time = time.time()
        try:
            # 1. Load document
            docs = DocumentLoaderFactory.load_file(file_path)
            if not docs:
                raise IngestionError(f"No text content could be extracted from {file_path}")
                
            # 2. Chunk document
            chunks = self.chunker.chunk_documents(docs)
            if not chunks:
                raise IngestionError("Document chunking produced 0 chunks.")

            # 3. Generate embeddings in batch
            texts = [c.text for c in chunks]
            embeddings = self.embedding_provider.embed_batch(texts)
            
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

            # 4. Store in vector database
            added_count = self.vector_store.add_chunks(chunks)

            duration = round(time.time() - start_time, 2)
            return {
                "file_path": file_path,
                "document_count": len(docs),
                "chunks_added": added_count,
                "embedding_provider": self.embedding_provider.provider_name,
                "duration_seconds": duration
            }
        except Exception as e:
            raise IngestionError(f"Ingestion pipeline failed for '{file_path}': {str(e)}")

    def ingest_documents(self, documents: List[Document]) -> int:
        """Ingest pre-constructed Document objects directly."""
        chunks = self.chunker.chunk_documents(documents)
        texts = [c.text for c in chunks]
        embeddings = self.embedding_provider.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        return self.vector_store.add_chunks(chunks)

    def chat(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> ChatResponse:
        """
        Execute full RAG generation loop:
        1. Retrieve top-K relevant chunks for query
        2. Format context and build prompt
        3. Call active LLM provider (Anthropic, Azure OpenAI, or Ollama)
        4. Return ChatResponse with citations and latency metrics
        """
        start_time = time.time()
        history = chat_history or []

        # 1. Retrieve relevant chunks
        search_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # 2. Format context
        formatted_context = self.retriever.format_context(search_results)

        # 3. Build prompt and conversation messages
        messages = RAGPromptBuilder.prepare_chat_messages(
            history=history,
            current_query=query,
            formatted_context=formatted_context
        )
        sys_prompt = RAGPromptBuilder.build_system_prompt(system_prompt)

        # 4. Call LLM
        try:
            llm_result = self.llm_provider.generate(
                messages=messages,
                system_prompt=sys_prompt,
                temperature=temperature
            )
        except Exception as e:
            raise LLMProviderError(f"RAG chat generation failed: {str(e)}")

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return ChatResponse(
            answer=llm_result["text"],
            sources=search_results,
            provider=llm_result["provider"],
            model_used=llm_result["model"],
            latency_ms=latency_ms,
            usage_info=llm_result.get("usage")
        )

    def get_knowledge_base_summary(self) -> Dict[str, Any]:
        """Return summary statistics of the indexed knowledge base."""
        return {
            "total_documents": len(self.vector_store.list_documents()),
            "total_chunks": self.vector_store.get_chunk_count(),
            "documents": self.vector_store.list_documents(),
            "active_llm_provider": self.llm_provider.provider_name,
            "active_model": self.llm_provider.current_model,
            "active_embedding_provider": self.embedding_provider.provider_name
        }
