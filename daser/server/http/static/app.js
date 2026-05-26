const state = {
  documents: [],
  selected: new Set(),
  details: new Map(),
};

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const detail = data && data.detail ? data.detail : response.statusText;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join("; ")
      : detail;
    throw new Error(message);
  }
  return data;
}

function setStatus(id, message, kind = "") {
  const element = document.getElementById(id);
  element.textContent = message;
  element.className = kind ? `status-line ${kind}` : "status-line";
}

function formatMs(value) {
  return `${Number(value || 0).toFixed(1)} ms`;
}

function shortKey(value) {
  return String(value || "").slice(0, 10);
}

function formatTime(value) {
  if (!value) {
    return "未知时间";
  }
  return new Date(Number(value) * 1000).toLocaleString("zh-CN");
}

function chunkRatio(doc) {
  if (
    Number.isFinite(Number(doc.chunk_count_cached)) &&
    Number.isFinite(Number(doc.chunk_count_total))
  ) {
    return `${Number(doc.chunk_count_cached)}/${Number(doc.chunk_count_total)}`;
  }
  const mask = Array.isArray(doc.cached_mask) ? doc.cached_mask : [];
  if (!mask.length) {
    return "0/0";
  }
  const cached = mask.filter(Boolean).length;
  return `${cached}/${mask.length}`;
}

async function loadHealth() {
  try {
    const health = await requestJson("/health");
    const healthText = document.getElementById("healthText");
    const vllmText = document.getElementById("vllmText");
    healthText.textContent = health.status === "ok" ? "服务可用" : "服务降级";
    healthText.className = health.status === "ok" ? "status-pill" : "status-pill error";
    vllmText.textContent = health.vllm ? "vLLM 已连接" : "vLLM 异常";
    vllmText.className = health.vllm ? "status-pill muted" : "status-pill error";
  } catch (error) {
    document.getElementById("healthText").textContent = "服务不可用";
    document.getElementById("healthText").className = "status-pill error";
    document.getElementById("vllmText").textContent = error.message;
    document.getElementById("vllmText").className = "status-pill error";
  }
}

async function loadDocuments() {
  const docs = await requestJson("/documents");
  state.documents = docs;
  const liveIds = new Set(docs.map((doc) => doc.doc_id));
  for (const docId of Array.from(state.selected)) {
    if (!liveIds.has(docId)) {
      state.selected.delete(docId);
    }
  }
  renderDocuments();
}

function renderDocuments() {
  const root = document.getElementById("documentsList");
  root.textContent = "";
  if (!state.documents.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "暂无文档，先上传一段文本。";
    root.appendChild(empty);
    return;
  }

  for (const doc of state.documents) {
    const card = document.createElement("article");
    card.className = "doc-card";

    const top = document.createElement("div");
    top.className = "doc-top";

    const title = document.createElement("label");
    title.className = "doc-title";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selected.has(doc.doc_id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        state.selected.add(doc.doc_id);
      } else {
        state.selected.delete(doc.doc_id);
      }
    });
    const titleText = document.createElement("span");
    titleText.textContent = doc.title || doc.doc_id;
    title.append(checkbox, titleText);

    const del = document.createElement("button");
    del.className = "delete-button";
    del.type = "button";
    del.textContent = "删除";
    del.addEventListener("click", () => deleteDocument(doc.doc_id));

    const actions = document.createElement("div");
    actions.className = "doc-actions";
    const view = document.createElement("button");
    view.className = "ghost-button";
    view.type = "button";
    view.textContent = "查看";
    view.addEventListener("click", () => showDocument(doc.doc_id));
    actions.append(view, del);

    top.append(title, actions);

    const meta = document.createElement("div");
    meta.className = "doc-meta";
    meta.innerHTML = `
      <span>状态：${doc.status || "unknown"}</span>
      <span>Token：${doc.token_count || 0}</span>
      <span>Chunk：${chunkRatio(doc)}</span>
      <span>${formatTime(doc.created_at)}</span>
    `;

    card.append(top, meta);
    root.appendChild(card);
  }
}

async function uploadDocument(event) {
  event.preventDefault();
  const title = document.getElementById("docTitle").value.trim();
  const text = document.getElementById("docText").value.trim();
  if (!title || !text) {
    setStatus("uploadStatus", "请输入标题和正文。", "error");
    return;
  }

  const button = event.submitter;
  button.disabled = true;
  setStatus("uploadStatus", "正在构建 KV Cache...", "");
  try {
    const result = await requestJson("/documents", {
      method: "POST",
      body: JSON.stringify({ title, text }),
    });
    setStatus(
      "uploadStatus",
      `构建完成：${result.chunk_count_cached}/${result.chunk_count} chunks，prefill ${formatMs(result.prefill_ms)}`,
      "ok",
    );
    state.selected.add(result.doc_id);
    document.getElementById("docTitle").value = "";
    document.getElementById("docText").value = "";
    await loadDocuments();
  } catch (error) {
    setStatus("uploadStatus", `上传失败：${error.message}`, "error");
  } finally {
    button.disabled = false;
  }
}

async function deleteDocument(docId) {
  try {
    await requestJson(`/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
    state.selected.delete(docId);
    state.details.delete(docId);
    await loadDocuments();
  } catch (error) {
    setStatus("uploadStatus", `删除失败：${error.message}`, "error");
  }
}

async function showDocument(docId) {
  try {
    const doc = state.details.get(docId)
      || await requestJson(`/documents/${encodeURIComponent(docId)}`);
    state.details.set(docId, doc);
    document.getElementById("documentPreviewTitle").textContent =
      doc.title || "文档详情";
    document.getElementById("documentPreviewMeta").textContent =
      `${doc.token_count || 0} tokens，${chunkRatio(doc)} chunks`;
    document.getElementById("documentPreviewText").textContent =
      doc.text || "该文档没有保存原文。";
  } catch (error) {
    document.getElementById("documentPreviewTitle").textContent = "加载失败";
    document.getElementById("documentPreviewMeta").textContent = "";
    document.getElementById("documentPreviewText").textContent = error.message;
  }
}

function renderMetrics(result) {
  const root = document.getElementById("metricsGrid");
  const hits = Array.isArray(result.cache_hits) ? result.cache_hits : [];
  const items = [
    ["模式", result.cache_enabled ? "使用 KV Cache" : "未使用 KV Cache"],
    ["延迟", formatMs(result.latency_ms)],
    ["Prompt Tokens", result.prompt_tokens || 0],
    ["Completion Tokens", result.completion_tokens || 0],
    ["Cache Hits", hits.length],
  ];
  root.textContent = "";
  for (const [label, value] of items) {
    const metric = document.createElement("div");
    metric.className = "metric";
    metric.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    root.appendChild(metric);
  }
}

function renderCacheHits(hits) {
  const root = document.getElementById("cacheHits");
  root.textContent = "";
  if (!hits.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "暂无命中信息。";
    root.appendChild(empty);
    return;
  }
  for (const hit of hits) {
    const card = document.createElement("div");
    card.className = "hit-card";
    card.innerHTML = `
      <strong>${shortKey(hit.chunk_key)}</strong>
      <div class="hit-meta">
        <span>tokens：${hit.token_count}</span>
        <span>target：${hit.target_token_start}</span>
        <span>pos offset：${hit.pos_offset}</span>
      </div>
    `;
    root.appendChild(card);
  }
}

async function runInference(event) {
  event.preventDefault();
  const docIds = Array.from(state.selected);
  const task = document.getElementById("taskText").value.trim();
  if (!docIds.length) {
    setStatus("inferStatus", "请至少选择一个文档。", "error");
    return;
  }
  if (!task) {
    setStatus("inferStatus", "请输入推理任务。", "error");
    return;
  }

  const button = event.submitter;
  button.disabled = true;
  setStatus("inferStatus", "推理中...", "");
  try {
    const useKvCache = document.getElementById("useKvCache").checked;
    const maxTokens = Number(document.getElementById("maxTokens").value || 80);
    const temperature = Number(document.getElementById("temperature").value || 0);
    const result = await requestJson("/infer", {
      method: "POST",
      body: JSON.stringify({
        doc_ids: docIds,
        task,
        use_kv_cache: useKvCache,
        gen_params: { max_tokens: maxTokens, temperature },
      }),
    });
    document.getElementById("resultMode").textContent = result.cache_enabled
      ? "已启用 KV Cache"
      : "未使用 KV Cache";
    document.getElementById("answerText").textContent = result.text || "模型未返回文本。";
    document.getElementById("promptPreviewText").textContent =
      result.prompt_preview || "服务端未返回 prompt 预览。";
    renderMetrics(result);
    renderCacheHits(Array.isArray(result.cache_hits) ? result.cache_hits : []);
    setStatus("inferStatus", "推理完成。", "ok");
  } catch (error) {
    setStatus("inferStatus", `推理失败：${error.message}`, "error");
  } finally {
    button.disabled = false;
  }
}

function bindEvents() {
  document.getElementById("uploadForm").addEventListener("submit", uploadDocument);
  document.getElementById("inferForm").addEventListener("submit", runInference);
  document.getElementById("refreshDocuments").addEventListener("click", loadDocuments);
}

async function boot() {
  bindEvents();
  await loadHealth();
  await loadDocuments();
}

boot().catch((error) => {
  setStatus("uploadStatus", `初始化失败：${error.message}`, "error");
});
