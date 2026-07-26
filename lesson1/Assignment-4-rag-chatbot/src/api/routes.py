"""
FastAPI Routes for RAG Chatbot.
Handles chat generation, document upload & indexing, and dynamic provider/endpoint switching.
"""

import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List, Dict, Any

from src.api.schemas import (
    ChatRequestSchema,
    ChatResponseSchema,
    SwitchProviderRequestSchema,
    KnowledgeBaseSummarySchema
)
from src.embeddings import EmbeddingProviderFactory
from src.storage import NumPyVectorStore
from src.llm import LLMProviderFactory
from src.llm.anthropic_client import AnthropicLLMProvider
from src.llm.azure_client import AzureOpenAILLMProvider
from src.llm.ollama_client import OllamaLLMProvider
from src.rag import RAGPipeline
from src.config.settings import settings

router = APIRouter(prefix="/api", tags=["RAG Chatbot"])

# Initialize Global Singleton Pipeline
vector_store = NumPyVectorStore(db_path=settings.vector_store_path)
embedding_provider = EmbeddingProviderFactory.get_provider()
llm_provider = LLMProviderFactory.get_provider()

rag_pipeline = RAGPipeline(
    llm_provider=llm_provider,
    embedding_provider=embedding_provider,
    vector_store=vector_store,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)


@router.post("/chat", response_model=ChatResponseSchema)
async def chat_endpoint(request: ChatRequestSchema):
    """
    Generate an augmented chat response using retrieved knowledge base documents.
    """
    try:
        # If user specified a temporary provider override in the request
        if request.provider and request.provider.lower() != rag_pipeline.llm_provider.provider_name:
            temp_provider = LLMProviderFactory.get_provider(request.provider)
            rag_pipeline.switch_llm_provider(temp_provider)

        response = rag_pipeline.chat(
            query=request.query,
            chat_history=request.chat_history,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            system_prompt=request.system_prompt,
            temperature=request.temperature or 0.3
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat generation failed: {str(e)}"
        )


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document_endpoint(file: UploadFile = File(...)):
    """
    Upload and index a new knowledge base document (PDF, Markdown, TXT, or CSV).
    """
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Move to persistent data/sample_docs or ingest directly
        dest_dir = "./data/uploaded_docs"
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, file.filename or "untitled.txt")
        shutil.copy2(tmp_path, dest_path)
        
        result = rag_pipeline.ingest_file(dest_path)
        return {
            "message": "Document indexed successfully",
            "details": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document ingestion failed: {str(e)}"
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/switch-provider")
async def switch_provider_endpoint(request: SwitchProviderRequestSchema):
    """
    Dynamically switch LLM provider or endpoints at runtime (e.g. Azure OpenAI endpoint switching or Anthropic proxy).
    """
    try:
        provider_type = request.provider.lower()
        if provider_type == "anthropic":
            if isinstance(rag_pipeline.llm_provider, AnthropicLLMProvider):
                rag_pipeline.llm_provider.update_proxy(
                    base_url=request.anthropic_base_url or settings.anthropic_base_url,
                    auth_token=request.anthropic_auth_token or settings.anthropic_auth_token or settings.anthropic_api_key or "",
                    model_name=request.anthropic_model or settings.anthropic_model
                )
            else:
                new_provider = AnthropicLLMProvider(
                    base_url=request.anthropic_base_url or settings.anthropic_base_url,
                    auth_token=request.anthropic_auth_token or settings.anthropic_auth_token or settings.anthropic_api_key,
                    model_name=request.anthropic_model or settings.anthropic_model
                )
                rag_pipeline.switch_llm_provider(new_provider)

        elif provider_type in ("azure_openai", "azure"):
            if isinstance(rag_pipeline.llm_provider, AzureOpenAILLMProvider) and request.azure_endpoint:
                rag_pipeline.llm_provider.update_endpoint(
                    new_endpoint=request.azure_endpoint,
                    new_api_key=request.azure_api_key,
                    new_deployment_name=request.azure_deployment_name,
                    new_api_version=request.azure_api_version
                )
            else:
                new_provider = AzureOpenAILLMProvider(
                    endpoint=request.azure_endpoint or settings.azure_openai_endpoint,
                    api_key=request.azure_api_key or settings.azure_openai_api_key,
                    api_version=request.azure_api_version or settings.azure_openai_api_version,
                    deployment_name=request.azure_deployment_name or settings.azure_openai_deployment_name
                )
                rag_pipeline.switch_llm_provider(new_provider)

        elif provider_type == "ollama":
            if isinstance(rag_pipeline.llm_provider, OllamaLLMProvider):
                rag_pipeline.llm_provider.update_model(
                    new_model_name=request.ollama_model or settings.ollama_model,
                    new_base_url=request.ollama_base_url or settings.ollama_base_url
                )
            else:
                new_provider = OllamaLLMProvider(
                    base_url=request.ollama_base_url or settings.ollama_base_url,
                    model_name=request.ollama_model or settings.ollama_model
                )
                rag_pipeline.switch_llm_provider(new_provider)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {request.provider}")

        return {
            "message": f"Successfully switched active LLM provider to '{provider_type}'",
            "active_provider": rag_pipeline.llm_provider.provider_name,
            "active_model": rag_pipeline.llm_provider.current_model
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider switching failed: {str(e)}"
        )


@router.get("/summary", response_model=KnowledgeBaseSummarySchema)
async def get_summary_endpoint():
    """Return summary statistics of the indexed knowledge base and active providers."""
    return rag_pipeline.get_knowledge_base_summary()


@router.delete("/documents/{doc_id}")
async def delete_document_endpoint(doc_id: str):
    """Delete a document and all its chunks from the vector database."""
    deleted_chunks = vector_store.delete_document(doc_id)
    if deleted_chunks == 0:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"message": f"Document '{doc_id}' deleted successfully", "chunks_removed": deleted_chunks}


@router.post("/clear-database")
async def clear_database_endpoint():
    """Clear all documents from the knowledge base."""
    vector_store.clear()
    return {"message": "Knowledge base cleared successfully."}
