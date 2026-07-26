# 🚀 Antigravity RAG Studio — Enterprise From-Scratch RAG Chatbot

An enterprise-grade, high-performance **Retrieval-Augmented Generation (RAG) chatbot** built **from scratch in Python without LangChain or LlamaIndex**. 

Designed with a clean, professional layered architecture, this system supports dynamic **Endpoint Switching** across **Azure OpenAI**, **Anthropic Claude (via custom proxy & auth token)**, and **Ollama (local offline LLMs)**.

---

## 🌟 Key Features & Architectural Highlights

### 1. 🏗️ True "From-Scratch" RAG Engine (No LangChain / LlamaIndex)
Instead of relying on bloated wrapper frameworks that obscure control flow and introduce heavy dependency chains, this system implements the fundamental RAG primitives directly:
* **Custom Document Loaders (`src/ingestion/loaders.py`)**: Direct extraction and metadata tagging for PDF (`pypdf`), Markdown, Plain Text, and CSV files.
* **Recursive Character Text Chunker (`src/ingestion/chunker.py`)**: A pure-Python recursive text splitting algorithm using a natural separator hierarchy (`\n\n`, `\n`, `. `, words) with configurable sliding window overlap and chunk metadata preservation.
* **Vectorized NumPy + SQLite Vector Database (`src/storage/numpy_vector_store.py`)**: Zero external daemon required (no Chroma, no FAISS, no Qdrant). Combines SQLite for ACID-compliant persistent metadata storage with a high-speed NumPy 2D array for normalized dot-product cosine similarity ($\text{similarity} = \frac{A \cdot B}{||A|| ||B||}$).

### 2. ⚡ Dynamic Endpoint Switching (Azure OpenAI & Claude Proxy)
Implementing reference patterns from Azure OpenAI and enterprise LLM gateways, the application allows live runtime switching across LLM providers without restarting the server or losing chat session state:
* **Anthropic Claude (Proxy Support)**: Fully configured to connect through custom enterprise LLM gateways (e.g., Presidio LLM Gateway at `https://proxy.llm-gateway.ready.presidio.com`) using `ANTHROPIC_BASE_URL` and custom `ANTHROPIC_AUTH_TOKEN` headers.
* **Azure OpenAI Switching Endpoints**: Seamlessly switch between different Azure OpenAI resource URLs, API keys, regions, and deployment names (`gpt-4o`, `gpt-35-turbo`) on the fly via REST endpoints or UI sidebar tabs.
* **Ollama Local LLMs**: Privacy-first offline inference and embeddings (`llama3.1`, `nomic-embed-text`) via local HTTP REST APIs.

### 3. 🎨 Stunning Glassmorphism Web Studio & Streamlit UI
* **Enterprise Web Application (`src/ui/web/`)**: An ultra-responsive, dark-mode glassmorphism HTML5/Vanilla CSS web app served directly by FastAPI. Features real-time drag-and-drop document indexing, interactive hyperparameter tuning sliders, and an expandable citation drawer that highlights exact retrieved vector chunks and cosine similarity scores.
* **Streamlit Studio Alternative (`src/ui/streamlit_app.py`)**: A secondary Python-native interactive UI for rapid testing and demonstrations.

---

## 📂 Layered Folder Structure

```text
Assignment-4-rag-chatbot/
├── README.md                      # Comprehensive documentation & architectural overview
├── .env.example                   # Template for API keys, proxy URLs, and hyperparameters
├── .gitignore                     # Excludes virtual environments, database files, and caches
├── requirements.txt               # Lightweight, minimal dependencies (No LangChain/LlamaIndex)
├── setup.sh                       # One-click environment setup and dependency installer
├── run_app.sh                     # One-click application launcher
├── app.py                         # FastAPI server entrypoint hosting REST API and Web UI
├── data/
│   ├── sample_docs/               # Reference architectural guides & Azure switching docs
│   └── uploaded_docs/             # Storage for dynamically indexed user documents
├── src/
│   ├── config/settings.py         # Pydantic environment configuration management
│   ├── core/                      # Domain types (Document, TextChunk, SearchResult) & Exceptions
│   ├── ingestion/                 # From-scratch document loaders & recursive text chunker
│   ├── embeddings/                # Abstract embedding interface (Local, Azure, Ollama)
│   ├── storage/                   # Custom NumPy + SQLite persistent vector database
│   ├── llm/                       # Claude (Proxy), Azure OpenAI, and Ollama LLM providers
│   ├── rag/                       # Document retriever, prompt builder, and RAG orchestrator
│   ├── api/                       # FastAPI REST routes and Pydantic request/response schemas
│   └── ui/                        # Web Studio (HTML/CSS/JS) and Streamlit application
└── tests/                         # Automated unit & integration test suite
```

---

## 🛠️ Quickstart Setup & Installation

### Step 1: Clone & Configure Environment
1. Open your terminal in the project directory.
2. Copy the example configuration file and add your credentials:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` to insert your **Anthropic Auth Token** or **Azure OpenAI credentials**:
   ```ini
   DEFAULT_LLM_PROVIDER=anthropic
   ANTHROPIC_BASE_URL="https://proxy.llm-gateway.ready.presidio.com"
   ANTHROPIC_AUTH_TOKEN="your-actual-api-key-here"
   ```
   *(Note: You can also configure these live directly inside the UI sidebar after launching!)*

### Step 2: Run Setup Script
Run the automated setup script to generate the Python virtual environment (`venv`) and install minimal required packages:
```bash
chmod +x setup.sh run_app.sh
./setup.sh
```

### Step 3: Launch the Studio
Start the backend server and Web Studio:
```bash
./run_app.sh
```
* Open your browser and navigate to: **http://localhost:8000**
* *Optional*: To run the Streamlit alternative UI instead, execute:
  ```bash
  ./venv/bin/streamlit run src/ui/streamlit_app.py
  ```

---

## 🧪 Running Automated Tests

Our comprehensive pytest suite verifies the from-scratch chunking algorithm, vector cosine matrix math, and end-to-end RAG pipeline without requiring live external API keys:
```bash
./venv/bin/pytest tests/ -v
```

---

## 📹 Demo Video Recording Guide (For Submission)

When recording your demo video submission, follow this simple 3-minute script to impress evaluators:

1. **Introduction & Architecture (30 sec)**:
   * Show the directory structure in your IDE or terminal. Highlight that there is **NO LangChain or LlamaIndex** in `requirements.txt`.
   * Explain how `NumPyVectorStore` computes cosine similarity using pure NumPy vector matrix math and persists metadata in SQLite.
2. **Live Document Ingestion (45 sec)**:
   * Open the Web UI at `http://localhost:8000`.
   * Drag and drop one of the sample files from `data/sample_docs/` (e.g., `azure_switching_endpoints_guide.md` or a custom PDF) into the sidebar dropzone.
   * Point out the chunk count updating in real time on the Knowledge Base badge.
3. **Q&A & Citation Inspection (45 sec)**:
   * Ask a question like: *"How do we switch endpoints dynamically in Azure OpenAI?"*
   * When the AI responds, click on the **"Retrieved Sources" citation drawer** to show the exact chunk snippet and cosine similarity score retrieved from the NumPy database.
4. **Live Endpoint Switching (40 sec)**:
   * In the sidebar under **"LLM Provider & Endpoints"**, switch from **Claude** to **Azure OpenAI** or **Ollama**.
   * Show how you can edit the endpoint URL or deployment name on the fly without restarting the app, successfully demonstrating the reference **Switching Endpoints pattern**!

---

## 📎 GitHub Submission Checklist

1. Ensure your `.env` file is listed in `.gitignore` so your personal API keys are never leaked!
2. Push your code to a public GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "feat: Build production RAG chatbot from scratch with endpoint switching"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```
3. Submit your public GitHub repo link and your demo recording link!

---
*Built with ❤️ using Clean Architecture and From-Scratch AI Engineering.*
