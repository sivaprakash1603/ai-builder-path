"""
Antigravity RAG Studio - Main Application Entrypoint.
Serves the REST API endpoints and the Enterprise AI Glassmorphism Web UI.
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.routes import router as api_router
from src.config.settings import settings

app = FastAPI(
    title="Antigravity RAG Studio",
    description="Enterprise RAG Chatbot built from scratch with Azure OpenAI Switching Endpoints, Claude Proxy, and Ollama.",
    version="1.0.0"
)

# Include REST API router
app.include_router(api_router)

# Mount static web directory for CSS and JS
static_dir = os.path.join(os.path.dirname(__file__), "src", "ui", "web")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serve the Enterprise RAG Studio web application interface."""
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        return {"message": "UI index.html not found. Please ensure src/ui/web/index.html exists."}
    return FileResponse(index_path)


if __name__ == "__main__":
    print("==============================================================================")
    print("🚀 Starting Antigravity RAG Studio (From-Scratch RAG Engine)")
    print(f"📡 Web Application & API available at: http://localhost:{settings.api_port}")
    print("==============================================================================")
    uvicorn.run("app:app", host=settings.api_host, port=settings.api_port, reload=True)
