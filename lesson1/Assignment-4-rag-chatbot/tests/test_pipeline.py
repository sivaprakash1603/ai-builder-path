"""
Integration tests for End-to-End RAG Pipeline.
"""

import os
import tempfile
import pytest
from typing import Dict, Any, List
from src.core.types import ChatMessage
from src.llm.base import BaseLLMProvider
from src.embeddings.local_embedding import LocalEmbeddingProvider
from src.storage.numpy_vector_store import NumPyVectorStore
from src.rag.pipeline import RAGPipeline


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for deterministic offline testing without API keys."""
    
    def __init__(self, model_name: str = "mock-model"):
        self._model_name = model_name

    def generate(
        self,
        messages: List[ChatMessage],
        system_prompt: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        last_msg = messages[-1].content if messages else ""
        return {
            "text": f"Mock response grounded in context for query: {last_msg[:30]}...",
            "model": self._model_name,
            "provider": "mock",
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}
        }

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def current_model(self) -> str:
        return self._model_name


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        db_path = tmp.name
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


def test_rag_pipeline_end_to_end(temp_db_path):
    vs = NumPyVectorStore(db_path=temp_db_path)
    emb = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    llm = MockLLMProvider()

    pipeline = RAGPipeline(
        llm_provider=llm,
        embedding_provider=emb,
        vector_store=vs,
        chunk_size=200,
        chunk_overlap=20
    )

    # Ingest sample document
    sample_file = "./data/sample_docs/azure_switching_endpoints_guide.md"
    if not os.path.exists(sample_file):
        pytest.skip("Sample document not found.")

    res = pipeline.ingest_file(sample_file)
    assert res["chunks_added"] > 0
    assert pipeline.vector_store.get_chunk_count() == res["chunks_added"]

    # Execute chat query
    query = "How do we switch endpoints dynamically in Azure OpenAI?"
    response = pipeline.chat(query=query, top_k=2)

    assert response.provider == "mock"
    assert response.model_used == "mock-model"
    assert len(response.sources) > 0
    assert response.latency_ms >= 0.0
    assert "Mock response" in response.answer
