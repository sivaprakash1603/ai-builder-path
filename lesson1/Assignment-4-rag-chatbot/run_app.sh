#!/usr/bin/env bash
# ==============================================================================
# Antigravity RAG Studio - Launch Script
# ==============================================================================

set -e

if [ ! -d "venv" ]; then
    echo "⚠️ venv not found! Running setup.sh first..."
    ./setup.sh
fi

echo "🚀 Launching Antigravity RAG Studio Server..."
echo "🌐 Open your browser at: http://localhost:8000"
echo "💡 To run Streamlit UI alternative instead, use: ./venv/bin/streamlit run src/ui/streamlit_app.py"
echo "=============================================================================="

./venv/bin/python app.py
