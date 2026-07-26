"""
Configuration management using Pydantic Settings.
Automatically loads environment variables from .env file or OS environment.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration settings with environment variable bindings."""
    
    # --- LLM Provider Selection ---
    default_llm_provider: str = Field("anthropic", description="'anthropic', 'azure_openai', or 'ollama'")

    # --- Anthropic (Claude) via Proxy or Direct ---
    anthropic_base_url: Optional[str] = Field("https://proxy.llm-gateway.ready.presidio.com", description="Proxy URL or standard Anthropic endpoint")
    anthropic_auth_token: Optional[str] = Field(None, description="Custom auth token for proxy gateway")
    anthropic_api_key: Optional[str] = Field(None, description="Standard Anthropic API key fallback")
    anthropic_model: str = Field("claude-3-5-sonnet-20241022", description="Model name for generation")

    # --- Azure OpenAI Configuration ---
    azure_openai_endpoint: Optional[str] = Field(None, description="Azure OpenAI resource endpoint URL")
    azure_openai_api_key: Optional[str] = Field(None, description="Azure OpenAI API key")
    azure_openai_api_version: str = Field("2024-08-01-preview", description="Azure OpenAI API version")
    azure_openai_deployment_name: str = Field("gpt-4o", description="Chat deployment name")
    azure_openai_embedding_deployment: str = Field("text-embedding-ada-002", description="Embedding deployment name")

    # --- Ollama (Local LLM & Embeddings) ---
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field("llama3.1:8b", description="Ollama LLM model name")
    ollama_embedding_model: str = Field("nomic-embed-text", description="Ollama embedding model name")

    # --- Embedding Provider Selection ---
    default_embedding_provider: str = Field("local", description="'local' (sentence-transformers), 'azure_openai', or 'ollama'")
    local_embedding_model: str = Field("all-MiniLM-L6-v2", description="HuggingFace model for local embeddings")

    # --- RAG Retrieval Settings ---
    chunk_size: int = Field(500, description="Target character count per chunk")
    chunk_overlap: int = Field(50, description="Character overlap between consecutive chunks")
    top_k_retrieval: int = Field(4, description="Number of top chunks to retrieve")
    similarity_threshold: float = Field(0.25, description="Minimum cosine similarity score to include in context")

    # --- Storage & Server Settings ---
    vector_store_path: str = Field("./data/vector_store.db", description="Path to persistent vector store")
    api_host: str = Field("0.0.0.0", description="FastAPI server host")
    api_port: int = Field(8000, description="FastAPI server port")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global settings instance singleton
settings = Settings()
