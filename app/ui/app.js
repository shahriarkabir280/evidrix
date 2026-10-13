// The Evidrix chat page: a thin client of the HTTP API (no framework, no build step).
//
// Text from the API is untrusted (it comes from documents and from a model that read them):
// it is set with textContent, except answers, which go through renderAnswer (escapes first).
import { abstentionMessage, formatSeconds, locationLabel, previewText, renderAnswer } from "./render.js";

const KEY_STORAGE = "evidrix.apiKey";
const JOB_POLL_MS = 2000;
const JOB_TIMEOUT_MS = 5 * 60 * 1000;

const $ = (id) => document.getElementById(id);
const state = {
  apiKey: readKey(),
  documents: [],
  selected: new Set(), // document ids that narrow the search; empty = all
  retrievalOnly: false, // the server answered /chat with chat_disabled
  busy: false,
};

class ApiError extends Error {
  constructor(status, body, headers) {
    const error = body?.error ?? {};
    super(error.message || `HTTP ${status}`);
    this.status = status;
    this.code = error.code ?? "http_error";
    this.requestId = error.request_id;
    this.retryAfter = headers.get("Retry-After");
  }
}

// --- API ---------------------------------------------------------------------------------

async function api(path, { method = "GET", json, form } = {}) {
  const headers = { Authorization: `Bearer ${state.apiKey}` };
  let body;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form) {
    body = form;
  }
  const response = await fetch(path, { method, headers, body });
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, data, response.headers);
  return { status: response.status, data };
}

function readKey() {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

function storeKey(key) {
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {
    // Private mode: the key lasts for this page only.
  }
}

// --- Small DOM helpers ---------------------------------------------------------------------

function el(tag, { className, text, attrs } = {}, ...children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  for (const [name, value] of Object.entries(attrs ?? {})) node.setAttribute(name, value);
  node.append(...children.filter(Boolean));
  return node;
}

function scrollToEnd() {
  const thread = $("thread");
  thread.scrollTop = thread.scrollHeight;
}

function describeError(error) {
  if (!(error instanceof ApiError)) return "Can't reach the server. Is the API running?";
  let text = error.message;
  if (error.code === "rate_limited" && error.retryAfter) text += ` Try again in ${error.retryAfter} s.`;
  return text;
}

// --- API key -------------------------------------------------------------------------------

async function askForKey(message) {
  const dialog = $("key-dialog");
  $("key-error").hidden = !message;
  $("key-error").textContent = message ?? "";
  $("key-input").value = "";
  dialog.showModal();
  await new Promise((resolve) => dialog.addEventListener("close", resolve, { once: true }));
  state.apiKey = $("key-input").value.trim();
  try {
    await api("/documents?limit=1");
    storeKey(state.apiKey);
  } catch (error) {
    const invalid = error instanceof ApiError && error.status === 401;
    return askForKey(invalid ? "That key was not accepted." : describeError(error));
  }
}

// --- Documents -----------------------------------------------------------------------------

async function loadDocuments() {
  const { data } = await api("/documents?limit=100");
  state.documents = data.items;
  renderDocuments();
}

function renderDocuments() {
  const list = $("documents");
  list.replaceChildren();
  for (const doc of state.documents) {
    const ready = doc.status === "completed";
    const box = el("input", { attrs: { type: "checkbox", "aria-label": `Search ${doc.title}` } });
    box.checked = state.selected.has(doc.id);
    box.disabled = !ready;
    box.addEventListener("change", () => {
      if (box.checked) state.selected.add(doc.id);
      else state.selected.delete(doc.id);
      renderScope();
    });
    const label = el("label", {}, box, el("span", { className: "name", text: doc.title, attrs: { title: doc.filename } }));
    const badge = ready ? null : el("span", { className: `badge ${doc.status}`, text: doc.status });
    if (doc.status === "failed" && doc.error_message) badge.title = doc.error_message;
    list.append(el("li", {}, label, badge));
  }
  if (!state.documents.length) list.append(el("li", { className: "hint", text: "No documents yet." }));
  renderScope();
}

function renderScope() {
  const n = state.selected.size;
  $("scope").textContent = n ? `${n} document${n > 1 ? "s" : ""} selected` : "All documents";
}

async function upload(file) {
  const form = new FormData();
  form.append("file", file);
  try {
    const { status, data } = await api("/documents", { method: "POST", form });
    await loadDocuments();
    if (status === 200 || !data.job_id) return; // already uploaded and processed
    await waitForJob(data.job_id);
  } catch (error) {
    alert(`Upload failed: ${describeError(error)}`);
  } finally {
    await loadDocuments().catch(() => {});
  }
}

async function waitForJob(jobId) {
  const deadline = Date.now() + JOB_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, JOB_POLL_MS));
    const { data: job } = await api(`/jobs/${jobId}`);
    if (job.status === "completed" || job.status === "failed") return job;
    await loadDocuments(); // shows queued → processing
  }
}

// --- History -------------------------------------------------------------------------------

async function loadHistory() {
  const { data } = await api("/chats?limit=50");
  const list = $("history");
  list.replaceChildren();
  for (const chat of data.items) {
    const button = el("button", {
      text: chat.question,
      attrs: { type: "button", title: chat.question, "data-chat": chat.id },
    });
    button.addEventListener("click", () => showChat(chat.id, button));
    list.append(el("li", {}, button));
  }
}

async function showChat(chatId, button) {
  document.querySelectorAll(".history button.active").forEach((b) => b.classList.remove("active"));
  button?.classList.add("active");
  closeSidebar();
  clearThread();
  try {
    const { data: chat } = await api(`/chats/${chatId}`);
    appendUser(chat.question);
    const container = appendAssistant();
    if (chat.error_code) {
      renderError(container, `This question failed (${chat.error_code}).`);
    } else {
      renderAnswerMessage(container, {
        answer: chat.answer,
        abstained: chat.abstained,
        abstain_reason: chat.abstain_reason,
        citations: chat.citations,
        retrieved: chat.retrieved, // stored without passage text
        timings_ms: chat.timings_ms,
        usage: { prompt_tokens: chat.prompt_tokens, completion_tokens: chat.completion_tokens },
        prompt_version: chat.prompt_version,
      });
    }
  } catch (error) {
    renderError(appendAssistant(), describeError(error));
  }
}

// --- Thread --------------------------------------------------------------------------------

function clearThread() {
  $("thread").replaceChildren();
}

function appendUser(question) {
  $("empty")?.remove();
  $("thread").append(
    el("div", { className: "message user" }, el("div", { className: "bubble", text: question })),
  );
  scrollToEnd();
}

function appendAssistant() {
  const container = el("div", { className: "message assistant" }, el("div", { className: "who", text: "Evidrix" }));
  $("thread").append(container);
  return container;
}

function renderThinking(container, label) {
  const timer = el("span", { text: "" });
  const node = el("div", { className: "thinking" }, el("span", { className: "dot" }), el("span", { text: label }), timer);
  container.append(node);
  const started = Date.now();
  const tick = setInterval(() => {
    timer.textContent = ` ${Math.round((Date.now() - started) / 1000)} s`;
  }, 1000);
  scrollToEnd();
  return () => {
    clearInterval(tick);
    node.remove();
  };
}

function renderError(container, message) {
  container.classList.add("error");
  container.append(el("div", { className: "bubble", text: message }));
  scrollToEnd();
}

function renderAnswerMessage(container, data) {
  const bubble = el("div", { className: "bubble" });
  if (data.abstained) {
    container.classList.add("abstained");
    bubble.textContent = abstentionMessage(data.abstain_reason);
  } else {
    bubble.innerHTML = renderAnswer(data.answer); // escaped by renderAnswer
  }
  container.append(bubble);

  // Citation buttons in the answer open the cited passage.
  const passages = new Map((data.retrieved ?? []).map((hit) => [hit.chunk_id, hit]));
  const byMarker = new Map(data.citations.map((c) => [String(c.marker), c]));
  bubble.addEventListener("click", (event) => {
    const marker = event.target.closest(".cite")?.dataset.cite;
    const citation = marker && byMarker.get(marker);
    if (citation) openSource(citation, passages.get(citation.chunk_id));
  });

  if (data.citations.length) {
    const sources = el("div", { className: "sources" });
    for (const citation of data.citations) {
      const card = el(
        "button",
        { className: "source", attrs: { type: "button" } },
        el(
          "div",
          { className: "head" },
          el("span", { className: "cite", text: String(citation.marker) }),
          el("span", { className: "title", text: citation.document_title }),
          el("span", { className: "location", text: locationLabel(citation) }),
        ),
        el("div", { className: "quote", text: previewText(citation.quote) }),
      );
      card.addEventListener("click", () => openSource(citation, passages.get(citation.chunk_id)));
      sources.append(card);
    }
    container.append(sources);
  }

  if (data.retrieved?.length) container.append(renderPassages(data.retrieved));

  const meta = [];
  if (data.timings_ms?.total != null) meta.push(formatSeconds(data.timings_ms.total));
  if (data.usage?.prompt_tokens) {
    meta.push(`${(data.usage.prompt_tokens + data.usage.completion_tokens).toLocaleString()} tokens`);
  }
  if (data.prompt_version) meta.push(data.prompt_version);
  if (meta.length) container.append(el("div", { className: "meta", text: meta.join(" · ") }));
  scrollToEnd();
}

function renderPassages(hits) {
  const list = el("ol");
  for (const hit of hits) {
    const head = el(
      "div",
      {},
      el("strong", { text: hit.document_title }),
      el("span", { className: "location", text: ` ${locationLabel(hit)}` }),
      el("span", { className: "score", text: ` · score ${hit.score.toFixed(3)}` }),
    );
    const item = el("li", {}, head);
    if (hit.text) item.append(el("div", { className: "quote", text: previewText(hit.text) }));
    list.append(item);
  }
  return el("details", { className: "passages" }, el("summary", { text: `Retrieved passages (${hits.length})` }), list);
}

function openSource(citation, passage) {
  $("source-title").textContent = `[${citation.marker}] ${citation.document_title}`;
  $("source-location").textContent = locationLabel(citation);
  $("source-text").textContent = passage?.text ?? citation.quote;
  $("source-dialog").showModal();
}

// --- Asking --------------------------------------------------------------------------------

async function ask(question) {
  const body = { question };
  if (state.selected.size) body.document_ids = [...state.selected];
  appendUser(question);
  const container = appendAssistant();

  if (!state.retrievalOnly) {
    const stop = renderThinking(container, "Searching your documents and writing an answer…");
    try {
      const { data } = await api("/chat", { method: "POST", json: body });
      stop();
      renderAnswerMessage(container, { ...data, prompt_version: data.config?.prompt_version });
      loadHistory().catch(() => {});
      return;
    } catch (error) {
      stop();
      if (!(error instanceof ApiError && error.code === "chat_disabled")) {
        renderError(container, describeError(error));
        return;
      }
      setRetrievalOnly(); // this deployment has no LLM: show the evidence instead
    }
  }

  const stop = renderThinking(container, "Searching your documents…");
  try {
    const { data } = await api("/retrieve", { method: "POST", json: body });
    stop();
    container.append(
      el("div", {
        className: "bubble",
        text: data.hits.length
          ? "This demo runs without a language model, so here are the passages an answer would be written from:"
          : "Nothing in your documents matches this question.",
      }),
    );
    if (data.hits.length) {
      const passages = renderPassages(data.hits);
      passages.open = true;
      container.append(passages);
    }
    scrollToEnd();
  } catch (error) {
    stop();
    renderError(container, describeError(error));
  }
}

function setRetrievalOnly() {
  state.retrievalOnly = true;
  $("mode").hidden = false;
  $("question").placeholder = "Search your documents…";
}

// --- Wiring --------------------------------------------------------------------------------

function closeSidebar() {
  $("sidebar").classList.remove("open");
}

function autosize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight}px`;
}

function wire() {
  const question = $("question");
  $("composer").addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = question.value.trim();
    if (!text || state.busy) return;
    state.busy = true;
    $("send").disabled = true;
    question.value = "";
    autosize(question);
    try {
      await ask(text);
    } finally {
      state.busy = false;
      $("send").disabled = false;
      question.focus();
    }
  });
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  question.addEventListener("input", () => autosize(question));

  $("new-chat").addEventListener("click", () => {
    document.querySelectorAll(".history button.active").forEach((b) => b.classList.remove("active"));
    clearThread();
    $("thread").append(
      el("div", { className: "empty", attrs: { id: "empty" } }, el("h1", { text: "Ask your documents" })),
    );
    closeSidebar();
    question.focus();
  });
  $("upload").addEventListener("change", (event) => {
    const file = event.target.files[0];
    event.target.value = "";
    if (file) upload(file);
  });
  $("sign-out").addEventListener("click", async () => {
    storeKey("");
    await askForKey();
    await refresh();
  });
  $("menu").addEventListener("click", () => $("sidebar").classList.toggle("open"));
}

async function refresh() {
  await Promise.all([loadDocuments(), loadHistory()]);
}

async function start() {
  wire();
  if (!state.apiKey) await askForKey();
  try {
    await refresh();
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) {
      await askForKey("Your saved key is no longer accepted.");
      await refresh();
    } else {
      renderError(appendAssistant(), describeError(error));
    }
  }
  $("question").focus();
}

start();
