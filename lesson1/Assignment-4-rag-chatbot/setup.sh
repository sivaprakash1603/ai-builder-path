#!/usr/bin/env bash
# ==============================================================================
# Antigravity RAG Studio - Environment Setup Script
# ==============================================================================

set -e

echo "🚀 Setting up Antigravity RAG Studio environment..."

if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment (venv)..."
    python3 -m venv venv
fi

echo "🔄 Upgrading pip and installing dependencies (No LangChain or LlamaIndex!)..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo "📁 Creating necessary data directories..."
mkdir -p data/sample_docs data/uploaded_docs

if [ ! -f ".env" ]; then
    echo "⚙️ Creating default .env from .env.example..."
    cp .env.example .env
    echo "💡 Note: Please update .env with your ANTHROPIC_AUTH_TOKEN or Azure OpenAI keys!"
fi

echo "✅ Setup complete! You can run the application with: ./run_app.sh"
