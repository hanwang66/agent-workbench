const sourceLangEl = document.getElementById("sourceLang");
const targetLangEl = document.getElementById("targetLang");
const sourceTextEl = document.getElementById("sourceText");
const resultTextEl = document.getElementById("resultText");
const sessionIdEl = document.getElementById("sessionId");
const styleEl = document.getElementById("style");
const domainEl = document.getElementById("domain");
const ragTopKEl = document.getElementById("ragTopK");
const retrievalModeEl = document.getElementById("retrievalMode");
const useRagEl = document.getElementById("useRag");
const ragEnabledViewEl = document.getElementById("ragEnabledView");
const ragTopKViewEl = document.getElementById("ragTopKView");
const retrievalModeViewEl = document.getElementById("retrievalModeView");
const ragUsedViewEl = document.getElementById("ragUsedView");
const ragChunksViewEl = document.getElementById("ragChunksView");
const glossaryTextEl = document.getElementById("glossaryText");
const ragTitleEl = document.getElementById("ragTitle");
const ragTextEl = document.getElementById("ragText");
const ragFileEl = document.getElementById("ragFile");
const ragDocsEl = document.getElementById("ragDocs");
const translateBtn = document.getElementById("translateBtn");
const ingestRagBtn = document.getElementById("ingestRagBtn");
const uploadRagBtn = document.getElementById("uploadRagBtn");
const refreshRagBtn = document.getElementById("refreshRagBtn");
const clearRagBtn = document.getElementById("clearRagBtn");
const clearSessionBtn = document.getElementById("clearSessionBtn");
const swapBtn = document.getElementById("swapBtn");
const statusEl = document.getElementById("status");
const ragHistoryBodyEl = document.getElementById("ragHistoryBody");

const ragHistory = [];
const MAX_RAG_HISTORY = 10;

function setStatus(text) {
  statusEl.textContent = text;
}

function getErrorMessage(data, fallback) {
  return data?.error?.message || data?.detail || data?.message || fallback;
}

function resetRagViews() {
  ragEnabledViewEl.value = "false";
  ragTopKViewEl.value = "-";
  retrievalModeViewEl.value = "-";
  ragUsedViewEl.value = "false";
  ragChunksViewEl.value = "";
}

function renderRagHistory() {
  if (!ragHistoryBodyEl) {
    return;
  }

  if (ragHistory.length === 0) {
    ragHistoryBodyEl.innerHTML = "<tr><td colspan=\"7\">暂无请求记录</td></tr>";
    return;
  }

  ragHistoryBodyEl.innerHTML = "";
  for (const item of ragHistory) {
    const row = document.createElement("tr");
    const columns = [item.time, item.summary, item.mode, item.useRag, item.topK, item.ragUsed, item.ragChunks];

    for (const value of columns) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.appendChild(cell);
    }
    ragHistoryBodyEl.appendChild(row);
  }
}

function appendRagHistory(entry) {
  ragHistory.unshift(entry);
  if (ragHistory.length > MAX_RAG_HISTORY) {
    ragHistory.pop();
  }
  renderRagHistory();
}

function parseGlossary(text) {
  const glossary = {};
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  for (const line of lines) {
    const separatorIndex = line.indexOf("=");
    if (separatorIndex <= 0 || separatorIndex === line.length - 1) {
      continue;
    }
    const key = line.slice(0, separatorIndex).trim();
    const value = line.slice(separatorIndex + 1).trim();
    if (key && value) {
      glossary[key] = value;
    }
  }
  return glossary;
}

async function translate() {
  const text = sourceTextEl.value.trim();
  const sourceLang = sourceLangEl.value.trim() || "Chinese";
  const targetLang = targetLangEl.value.trim() || "English";
  const style = styleEl.value.trim() || "neutral";
  const domain = domainEl.value.trim() || "general";
  const ragTopK = Number.parseInt(ragTopKEl.value, 10) || 3;
  const retrievalMode = (retrievalModeEl.value || "hybrid").trim().toLowerCase();
  const useRag = Boolean(useRagEl.checked);
  const sessionId = sessionIdEl.value.trim() || null;
  const glossary = parseGlossary(glossaryTextEl.value);
  const summary = text.length > 36 ? `${text.slice(0, 36)}...` : text;

  if (!text) {
    setStatus("请输入要翻译的内容");
    sourceTextEl.focus();
    return;
  }

  translateBtn.disabled = true;
  setStatus("翻译中...");
  ragEnabledViewEl.value = String(useRag);
  ragTopKViewEl.value = String(ragTopK);
  retrievalModeViewEl.value = retrievalMode;
  ragUsedViewEl.value = "false";
  ragChunksViewEl.value = "";

  try {
    const response = await fetch("/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        source_lang: sourceLang,
        target_lang: targetLang,
        style,
        domain,
        session_id: sessionId,
        glossary,
        use_rag: useRag,
        rag_top_k: ragTopK,
        retrieval_mode: retrievalMode,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(getErrorMessage(data, "翻译失败"));
    }

    resultTextEl.value = data.translated_text || "";
    if (data.session_id) {
      sessionIdEl.value = data.session_id;
    }
    ragUsedViewEl.value = String(Boolean(data.rag_used));
    ragChunksViewEl.value = Array.isArray(data.rag_chunks) && data.rag_chunks.length > 0
      ? data.rag_chunks.join("\n")
      : "无";
    appendRagHistory({
      time: new Date().toLocaleTimeString(),
      summary,
      mode: retrievalMode,
      useRag: String(useRag),
      topK: ragTopK,
      ragUsed: String(Boolean(data.rag_used)),
      ragChunks: Array.isArray(data.rag_chunks) && data.rag_chunks.length > 0 ? data.rag_chunks.join(", ") : "无",
    });
    const ragLabel = data.rag_used ? ` | RAG: ${data.rag_chunks.join(", ") || "hit"}` : " | RAG: off/no-hit";
    setStatus(`完成 (${data.model}) | 记忆轮次: ${data.memory_turns}${ragLabel}`);
  } catch (error) {
    appendRagHistory({
      time: new Date().toLocaleTimeString(),
      summary,
      mode: retrievalMode,
      useRag: String(useRag),
      topK: ragTopK,
      ragUsed: "error",
      ragChunks: "请求失败",
    });
    resetRagViews();
    setStatus(`错误: ${error.message}`);
  } finally {
    translateBtn.disabled = false;
  }
}

async function refreshRagDocs() {
  try {
    const response = await fetch("/rag/documents");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(getErrorMessage(data, "获取 RAG 文档失败"));
    }

    if (!Array.isArray(data) || data.length === 0) {
      ragDocsEl.value = "当前还没有 RAG 文档";
      return;
    }

    ragDocsEl.value = data
      .map((item) => `${item.doc_id} | ${item.title} | chunks=${item.chunks}`)
      .join("\n");
  } catch (error) {
    setStatus(`错误: ${error.message}`);
  }
}

async function ingestRagDocument() {
  const text = ragTextEl.value.trim();
  const title = ragTitleEl.value.trim() || "Untitled";

  if (!text) {
    setStatus("请先填写 RAG 文档内容");
    ragTextEl.focus();
    return;
  }

  ingestRagBtn.disabled = true;
  setStatus("RAG 文档入库中...");
  try {
    const response = await fetch("/rag/documents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, text }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(getErrorMessage(data, "RAG 入库失败"));
    }

    setStatus(`RAG 入库完成: ${data.title} (${data.chunks} chunks)`);
    await refreshRagDocs();
  } catch (error) {
    setStatus(`错误: ${error.message}`);
  } finally {
    ingestRagBtn.disabled = false;
  }
}

async function uploadRagDocument() {
  const file = ragFileEl.files && ragFileEl.files[0];
  if (!file) {
    setStatus("请先选择要上传的文档");
    return;
  }

  const title = ragTitleEl.value.trim();
  const formData = new FormData();
  formData.append("file", file);
  if (title) {
    formData.append("title", title);
  }

  uploadRagBtn.disabled = true;
  setStatus("文档上传并入库中...");
  try {
    const response = await fetch("/rag/documents/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(getErrorMessage(data, "文档上传失败"));
    }

    setStatus(`上传入库完成: ${data.title} (${data.chunks} chunks)`);
    ragFileEl.value = "";
    await refreshRagDocs();
  } catch (error) {
    setStatus(`错误: ${error.message}`);
  } finally {
    uploadRagBtn.disabled = false;
  }
}

async function clearRagDocuments() {
  clearRagBtn.disabled = true;
  try {
    const response = await fetch("/rag/documents", { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(getErrorMessage(data, "清空 RAG 文档失败"));
    }

    setStatus(`已清空 RAG 文档: ${data.deleted_documents}`);
    await refreshRagDocs();
  } catch (error) {
    setStatus(`错误: ${error.message}`);
  } finally {
    clearRagBtn.disabled = false;
  }
}

async function clearSession() {
  const sessionId = sessionIdEl.value.trim();
  if (!sessionId) {
    setStatus("当前没有会话 ID");
    return;
  }

  clearSessionBtn.disabled = true;
  try {
    const response = await fetch(`/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(getErrorMessage(data, "清理会话失败"));
    }
    setStatus(data.cleared ? "会话记忆已清空" : "会话不存在，已无记忆");
  } catch (error) {
    setStatus(`错误: ${error.message}`);
  } finally {
    clearSessionBtn.disabled = false;
  }
}

function swapLanguages() {
  const from = sourceLangEl.value;
  sourceLangEl.value = targetLangEl.value;
  targetLangEl.value = from;
}

translateBtn.addEventListener("click", translate);
ingestRagBtn.addEventListener("click", ingestRagDocument);
uploadRagBtn.addEventListener("click", uploadRagDocument);
refreshRagBtn.addEventListener("click", refreshRagDocs);
clearRagBtn.addEventListener("click", clearRagDocuments);
clearSessionBtn.addEventListener("click", clearSession);
swapBtn.addEventListener("click", swapLanguages);
sourceTextEl.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    translate();
  }
});

refreshRagDocs();
resetRagViews();
renderRagHistory();
