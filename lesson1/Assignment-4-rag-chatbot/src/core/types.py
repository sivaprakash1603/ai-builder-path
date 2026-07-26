"""
Core data models and type definitions for the RAG chatbot system.
Built using Pydantic for strict schema validation and serialization.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
import uuid
import time


class Document(BaseModel):
    """Represents a raw ingested knowledge document."""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = Field(..., description="File path or URL of the document")
    title: str = Field(..., description="Document title or filename")
    content: str = Field(..., description="Raw text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")


class TextChunk(BaseModel):
    """Represents a text segment produced by the chunking engine."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = Field(..., description="ID of the parent document")
    text: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Sequential index of the chunk in the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Combined document and chunk metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding representation")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SearchResult(BaseModel):
    """Represents a retrieved chunk with similarity scoring."""
    chunk_id: str
    doc_id: str
    text: str
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation."""
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")
    timestamp: float = Field(default_factory=time.time)


class ChatResponse(BaseModel):
    """Represents the complete response returned to the UI / client."""
    answer: str = Field(..., description="Generated answer from LLM")
    sources: List[SearchResult] = Field(default_factory=list, description="Retrieved chunks cited in answer")
    provider: str = Field(..., description="LLM provider used (anthropic, azure_openai, ollama)")
    model_used: str = Field(..., description="Specific model deployment or identifier")
    latency_ms: float = Field(default=0.0, description="Total generation time in milliseconds")
    usage_info: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")
