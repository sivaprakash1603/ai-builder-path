/* ==========================================================================
   RAG Chatbot - Frontend Logic & API Client
   ========================================================================== */

let chatHistory = [];
let activeProvider = "anthropic";

// Initialize on load
document.addEventListener("DOMContentLoaded", () => {
  fetchSummary();
  setupDragAndDrop();
});

/* --- Provider Tab Switching --- */
function switchTab(providerName) {
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(tab => tab.classList.remove("active"));
  
  const targetId = providerName === "anthropic" ? "tab-anthropic" : (providerName === "azure_openai" ? "tab-azure" : "tab-ollama");
  document.querySelector(`[data-target="${targetId}"]`).classList.add("active");
  document.getElementById(targetId).classList.add("active");
  activeProvider = providerName;
}

/* --- Apply Provider / Endpoint Switching --- */
async function applyProviderSwitch(provider) {
  const payload = { provider: provider };

  if (provider === "anthropic") {
    payload.anthropic_base_url = document.getElementById("anthropic-base-url").value;
    payload.anthropic_auth_token = document.getElementById("anthropic-auth-token").value;
    payload.anthropic_model = document.getElementById("anthropic-model").value;
  } else if (provider === "azure_openai") {
    payload.azure_endpoint = document.getElementById("azure-endpoint").value;
    payload.azure_api_key = document.getElementById("azure-api-key").value;
    payload.azure_deployment_name = document.getElementById("azure-deployment").value;
    payload.azure_api_version = document.getElementById("azure-version").value;
  } else if (provider === "ollama") {
    payload.ollama_base_url = document.getElementById("ollama-url").value;
    payload.ollama_model = document.getElementById("ollama-model").value;
  }

  try {
    const res = await fetch("/api/switch-provider", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (res.ok) {
      document.getElementById("active-provider-label").innerText = `Provider: ${data.active_provider} (${data.active_model})`;
      addSystemNotification(`✅ Successfully switched LLM Endpoint to <b>${data.active_provider}</b> [Model: ${data.active_model}]`);
      fetchSummary();
    } else {
      alert(`Failed to switch provider: ${data.detail || "Unknown error"}`);
    }
  } catch (err) {
    alert(`Network error during provider switch: ${err.message}`);
  }
}

/* --- Knowledge Base Summary & Document Management --- */
async function fetchSummary() {
  try {
    const res = await fetch("/api/summary");
    const data = await res.json();
    if (res.ok) {
      document.getElementById("kb-stats-badge").innerText = `${data.total_chunks} Chunks (${data.total_documents} Docs)`;
      document.getElementById("active-provider-label").innerText = `Provider: ${data.active_llm_provider} (${data.active_model})`;
      
      const listEl = document.getElementById("doc-list");
      if (!data.documents || data.documents.length === 0) {
        listEl.innerHTML = '<div style="font-size:11px; color:var(--text-muted); text-align:center; padding:10px;">No documents indexed yet. Upload one above!</div>';
      } else {
        listEl.innerHTML = data.documents.map(doc => `
          <div class="doc-item">
            <span class="doc-name" title="${doc.title}">${doc.title}</span>
            <span class="doc-badge">${doc.chunk_count} chunks</span>
            <button class="doc-delete-btn" onclick="deleteDocument('${doc.doc_id}')" title="Delete Document">×</button>
          </div>
        `).join("");
      }
    }
  } catch (err) {
    console.error("Failed to fetch KB summary:", err);
  }
}

async function deleteDocument(docId) {
  if (!confirm("Are you sure you want to delete this document from the knowledge base?")) return;
  try {
    const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
    if (res.ok) {
      fetchSummary();
      addSystemNotification("🗑️ Document removed from vector database.");
    }
  } catch (err) {
    alert("Error deleting document.");
  }
}

async function clearKnowledgeBase() {
  if (!confirm("WARNING: This will delete all indexed documents and clear the vector database. Proceed?")) return;
  try {
    const res = await fetch("/api/clear-database", { method: "POST" });
    if (res.ok) {
      fetchSummary();
      addSystemNotification("🧹 Knowledge base vector database cleared completely.");
    }
  } catch (err) {
    alert("Error clearing database.");
  }
}

/* --- Drag & Drop Document Upload --- */
function setupDragAndDrop() {
  const dropzone = document.getElementById("dropzone");
  
  ["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, preventDefaults, false);
  });
  
  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }
  
  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, () => dropzone.classList.add("dragover"), false);
  });
  
  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, () => dropzone.classList.remove("dragover"), false);
  });
  
  dropzone.addEventListener("drop", e => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFileUpload(files);
  }, false);
}

async function handleFileUpload(files) {
  if (!files || files.length === 0) return;
  const file = files[0];
  
  const dropzoneText = document.querySelector(".dropzone-text");
  const origText = dropzoneText.innerText;
  dropzoneText.innerText = `⏳ Indexing "${file.name}"...`;
  
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (res.ok) {
      addSystemNotification(`📄 Successfully indexed <b>${file.name}</b> (${data.details.chunks_added} vector chunks created!)`);
      fetchSummary();
    } else {
      alert(`Upload failed: ${data.detail || "Unknown error"}`);
    }
  } catch (err) {
    alert(`Network error during upload: ${err.message}`);
  } finally {
    dropzoneText.innerText = origText;
  }
}

/* --- Chat Generation & UI Rendering --- */
async function handleChatSubmit(event) {
  event.preventDefault();
  const inputEl = document.getElementById("chat-input-field");
  const query = inputEl.value.trim();
  if (!query) return;

  // Clear input
  inputEl.value = "";

  // Append user message to UI
  appendMessage("user", query);
  
  // Show loading indicator
  const loadingId = appendLoadingIndicator();

  // Gather hyperparameters
  const topK = parseInt(document.getElementById("top-k-slider").value, 10);
  const threshold = parseFloat(document.getElementById("thresh-slider").value);
  const temperature = parseFloat(document.getElementById("temp-slider").value);

  const payload = {
    query: query,
    chat_history: chatHistory.slice(-6), // Keep last 3 turns of context
    top_k: topK,
    similarity_threshold: threshold,
    temperature: temperature
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    
    // Remove loading indicator
    document.getElementById(loadingId)?.remove();

    const data = await res.json();
    if (res.ok) {
      // Append assistant answer with citations
      appendAssistantMessage(data.answer, data.sources, data.latency_ms, data.provider, data.model_used);
      
      // Update history
      chatHistory.push({ role: "user", content: query });
      chatHistory.push({ role: "assistant", content: data.answer });
    } else {
      appendMessage("assistant", `❌ Error generating response: ${data.detail || "Unknown error"}`);
    }
  } catch (err) {
    document.getElementById(loadingId)?.remove();
    appendMessage("assistant", `❌ Network error communicating with RAG server: ${err.message}`);
  }
}

function appendMessage(role, text) {
  const container = document.getElementById("chat-messages");
  const msgEl = document.createElement("div");
  msgEl.className = `message ${role}`;
  
  const avatar = role === "user" ? "You" : "AI";
  msgEl.innerHTML = `
    <div class="avatar">${avatar}</div>
    <div class="msg-bubble">${formatMarkdownText(text)}</div>
  `;
  
  container.appendChild(msgEl);
  container.scrollTop = container.scrollHeight;
}

function appendAssistantMessage(answer, sources, latency, provider, model) {
  const container = document.getElementById("chat-messages");
  const msgEl = document.createElement("div");
  msgEl.className = "message assistant";
  
  let citationsHtml = "";
  if (sources && sources.length > 0) {
    citationsHtml = `
      <div class="citations-wrapper">
        <div class="citations-title">🔍 Retrieved Sources (${sources.length} chunks cited | Latency: ${latency}ms)</div>
        ${sources.map((src, i) => `
          <div class="citation-card" onclick="this.classList.toggle('expanded')">
            <div class="citation-header">
              <span>[Source ${i+1}] ${src.metadata?.title || src.doc_id}</span>
              <span class="citation-score">Sim Score: ${src.score}</span>
            </div>
            <div class="citation-body">${src.text}</div>
          </div>
        `).join("")}
      </div>
    `;
  } else {
    citationsHtml = `<div style="margin-top:12px; font-size:11px; color:var(--text-muted);">⚡ Answered from model knowledge (0 chunks met similarity threshold | Latency: ${latency}ms)</div>`;
  }

  msgEl.innerHTML = `
    <div class="avatar">AI</div>
    <div class="msg-bubble">
      <div>${formatMarkdownText(answer)}</div>
      ${citationsHtml}
    </div>
  `;
  
  container.appendChild(msgEl);
  container.scrollTop = container.scrollHeight;
}

function appendLoadingIndicator() {
  const container = document.getElementById("chat-messages");
  const id = "loading-" + Date.now();
  const msgEl = document.createElement("div");
  msgEl.id = id;
  msgEl.className = "message assistant";
  msgEl.innerHTML = `
    <div class="avatar">AI</div>
    <div class="msg-bubble">
      <div class="loading-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  container.appendChild(msgEl);
  container.scrollTop = container.scrollHeight;
  return id;
}

function addSystemNotification(htmlText) {
  const container = document.getElementById("chat-messages");
  const notifEl = document.createElement("div");
  notifEl.style.cssText = "align-self: center; background: rgba(0, 245, 212, 0.1); border: 1px solid rgba(0, 245, 212, 0.3); color: var(--accent-cyan); padding: 8px 16px; border-radius: 20px; font-size: 12px; font-weight: 500; max-width: 80%; text-align: center; animation: fadeIn 0.4s;";
  notifEl.innerHTML = htmlText;
  container.appendChild(notifEl);
  container.scrollTop = container.scrollHeight;
}

function formatMarkdownText(text) {
  if (!text) return "";
  // Basic safe HTML formatting for newlines, bold, and bullet points
  return text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/\n\n/g, "<br><br>")
    .replace(/\n- /g, "<br>• ")
    .replace(/\n/g, "<br>");
}
