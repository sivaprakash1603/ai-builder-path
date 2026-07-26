"""
FastAPI Request and Response schemas for the RAG Chatbot REST API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.core.types import SearchResult, ChatMessage


class ChatRequestSchema(BaseModel):
    query: str = Field(..., description="User chat question")
    chat_history: Optional[List[ChatMessage]] = Field(default=None, description="Previous conversation turns")
    top_k: Optional[int] = Field(default=None, description="Override default top-K retrieval")
    similarity_threshold: Optional[float] = Field(default=None, description="Override similarity threshold")
    temperature: Optional[float] = Field(default=0.3, description="LLM generation temperature")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt override")
    provider: Optional[str] = Field(default=None, description="Target LLM provider (anthropic, azure_openai, ollama)")


class ChatResponseSchema(BaseModel):
    answer: str
    sources: List[SearchResult]
    provider: str
    model_used: str
    latency_ms: float
    usage_info: Optional[Dict[str, int]] = None


class SwitchProviderRequestSchema(BaseModel):
    provider: str = Field(..., description="'anthropic', 'azure_openai', or 'ollama'")
    # Anthropic settings
    anthropic_base_url: Optional[str] = None
    anthropic_auth_token: Optional[str] = None
    anthropic_model: Optional[str] = None
    # Azure OpenAI settings
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_deployment_name: Optional[str] = None
    azure_api_version: Optional[str] = None
    # Ollama settings
    ollama_base_url: Optional[str] = None
    ollama_model: Optional[str] = None


class KnowledgeBaseSummarySchema(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[Dict[str, Any]]
    active_llm_provider: str
    active_model: str
    active_embedding_provider: str
