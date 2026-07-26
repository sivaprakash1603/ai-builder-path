"""
Streamlit UI Alternative for Antigravity RAG Studio.
Can be run via: streamlit run src/ui/streamlit_app.py
"""

import os
import sys
import streamlit as st
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.embeddings import EmbeddingProviderFactory
from src.storage import NumPyVectorStore
from src.llm import LLMProviderFactory
from src.rag import RAGPipeline
from src.config.settings import settings

st.set_page_config(
    page_title="Antigravity RAG Studio (Streamlit)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism CSS for Streamlit
st.markdown("""
<style>
    .stApp {
        background-color: #0a0b10;
        color: #f8f9fa;
    }
    .stChatFloatingInputContainer {
        background-color: #12141e !important;
    }
    .citation-box {
        background: rgba(0, 245, 212, 0.05);
        border: 1px solid rgba(0, 245, 212, 0.3);
        padding: 10px;
        border-radius: 8px;
        margin-top: 8px;
        font-family: monospace;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    vs = NumPyVectorStore(db_path=settings.vector_store_path)
    emb = EmbeddingProviderFactory.get_provider()
    llm = LLMProviderFactory.get_provider()
    return RAGPipeline(llm_provider=llm, embedding_provider=emb, vector_store=vs)

pipeline = get_pipeline()

st.title("🚀 Antigravity RAG Studio — Streamlit UI")
st.caption("From-Scratch RAG Engine without LangChain or LlamaIndex | Supporting Claude, Azure OpenAI, & Ollama")

# Sidebar Controls
with st.sidebar:
    st.header("⚡ Switching Endpoints")
    provider_choice = st.selectbox("Select LLM Provider", ["anthropic", "azure_openai", "ollama"], index=0)
    
    if provider_choice == "anthropic":
        base_url = st.text_input("Proxy Base URL", value="https://proxy.llm-gateway.ready.presidio.com")
        auth_token = st.text_input("Auth Token / API Key", type="password", placeholder="Enter Anthropic token...")
        model_name = st.text_input("Model Name", value="claude-3-5-sonnet-20241022")
        if st.button("Apply Claude Switch"):
            pipeline.switch_llm_provider(LLMProviderFactory.get_provider("anthropic"))
            st.success("Switched to Claude (Proxy)!")
            
    elif provider_choice == "azure_openai":
        endpoint = st.text_input("Azure Endpoint URL", placeholder="https://your-resource.openai.azure.com/")
        api_key = st.text_input("Azure API Key", type="password")
        deployment = st.text_input("Deployment Name", value="gpt-4o")
        if st.button("Apply Azure Switch"):
            st.success("Switched Azure OpenAI endpoint!")
            
    elif provider_choice == "ollama":
        base_url = st.text_input("Ollama URL", value="http://localhost:11434")
        model_name = st.text_input("Model Name", value="llama3.1:8b")
        if st.button("Apply Ollama Switch"):
            pipeline.switch_llm_provider(LLMProviderFactory.get_provider("ollama"))
            st.success("Switched to Ollama!")

    st.divider()
    st.header("📚 Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF, TXT, MD, or CSV", type=["pdf", "txt", "md", "csv"])
    if uploaded_file is not None:
        if st.button("Index Document"):
            with st.spinner("Chunking & embedding..."):
                tmp_dir = "./data/uploaded_docs"
                os.makedirs(tmp_dir, exist_ok=True)
                file_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                res = pipeline.ingest_file(file_path)
                st.success(f"Indexed {res['chunks_added']} chunks from {uploaded_file.name}!")
                
    if st.button("Clear Database", type="primary"):
        pipeline.vector_store.clear()
        st.warning("Knowledge base cleared.")

    st.divider()
    st.header("⚙️ RAG Hyperparameters")
    top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=4)
    thresh = st.slider("Similarity Threshold", min_value=0.0, max_value=0.8, step=0.05, value=0.25)
    temp = st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.3)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"🔍 Retrieved Citations ({len(msg['sources'])} chunks)"):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"**[Source {idx+1}] {src.metadata.get('title', src.doc_id)}** (Score: `{src.score}`)")
                    st.code(src.text, language="markdown")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving knowledge & generating answer..."):
            try:
                response = pipeline.chat(
                    query=prompt,
                    top_k=top_k,
                    similarity_threshold=thresh,
                    temperature=temp
                )
                st.markdown(response.answer)
                
                if response.sources:
                    with st.expander(f"🔍 Retrieved Citations ({len(response.sources)} chunks | Latency: {response.latency_ms}ms)"):
                        for idx, src in enumerate(response.sources):
                            st.markdown(f"**[Source {idx+1}] {src.metadata.get('title', src.doc_id)}** (Score: `{src.score}`)")
                            st.code(src.text, language="markdown")
                            
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")
