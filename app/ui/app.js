// The Evidrix chat page: a thin client of the HTTP API (no framework, no build step).
//
// Text from the API is untrusted (it comes from documents and from a model that read them):
// it is set with textContent, except answers, which go through renderAnswer (escapes first).
import {
  abstentionMessage,
  formatSeconds,
  locationLabel,
  plainAnswer,
  previewText,
  renderAnswer,
  renderPassage,
} from "./render.js";

const KEY_STORAGE = "evidrix.apiKey";
const JOB_POLL_MS = 2000;
const JOB_TIMEOUT_MS = 5 * 60 * 1000;
const NARROW = window.matchMedia("(max-width: 860px)");

const $ = (id) => document.getElementById(id);
const state = {
  apiKey: readKey(),
  documents: [],
  scope: new Map(), // document id -> { title, status }: documents a question is limited to
  conversationId: null, // set by the first answer; sent with follow-ups
  retrievalOnly: false, // the server answered /chat with chat_disabled
  busy: false,
};

class ApiError extends Error {
  constructor(status, body, headers) {
    const error = body?.error ?? {};
    super(error.message || `HTTP ${status}`);
    this.status = status;
    this.code = error.code ?? "http_error";
    this.retryAfter = headers.get("Retry-After");
  }
}

// --- Icons (inline SVG: nothing to load, nothing for the CSP to allow) ----------------------

const ICONS = {
  sidebar: [["rect", { x: 3, y: 3, width: 18, height: 18, rx: 3 }], ["path", { d: "M9 3v18" }]],
  edit: [["path", { d: "M12 20h9" }], ["path", { d: "M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" }]],
  files: [["path", { d: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" }], ["path", { d: "M14 2v6h6" }]],
  key: [["circle", { cx: 7.5, cy: 15.5, r: 5.5 }], ["path", { d: "m21 2-9.6 9.6" }], ["path", { d: "m15.5 7.5 3 3L22 7l-3-3" }]],
  plus: [["path", { d: "M12 5v14" }], ["path", { d: "M5 12h14" }]],
  "arrow-up": [["path", { d: "M12 19V5" }], ["path", { d: "m5 12 7-7 7 7" }]],
  close: [["path", { d: "M18 6 6 18" }], ["path", { d: "m6 6 12 12" }]],
  clip: [["path", { d: "m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" }]],
  copy: [["rect", { x: 9, y: 9, width: 13, height: 13, rx: 2 }], ["path", { d: "M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" }]],
  check: [["path", { d: "M20 6 9 17l-5-5" }]],
  book: [["path", { d: "M4 19.5A2.5 2.5 0 0 1 6.5 17H20" }], ["path", { d: "M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" }]],
  search: [["circle", { cx: 11, cy: 11, r: 7 }], ["path", { d: "m20 20-3.5-3.5" }]],
};

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  for (const [k, v] of Object.entries({ viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" })) {
    svg.setAttribute(k, v);
  }
  for (const [tag, attrs] of ICONS[name]) {
    const part = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) part.setAttribute(k, v);
    svg.append(part);
  }
  return svg;
}

function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((node) => {
    if (!node.querySelector("svg")) node.prepend(icon(node.dataset.icon));
  });
}

// --- API -----------------------------------------------------------------------------------

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

function describeError(error) {
  if (!(error instanceof ApiError)) return "Can't reach the server. Is the API running?";
  let text = error.message;
  if (error.code === "rate_limited" && error.retryAfter) text += ` Try again in ${error.retryAfter} s.`;
  return text;
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

function iconButton(name, label, onClick) {
  const button = el("button", { className: "icon-btn", attrs: { type: "button", "aria-label": label, title: label } }, icon(name));
  button.addEventListener("click", onClick);
  return button;
}

function scrollToEnd() {
  const scroll = $("scroll");
  scroll.scrollTop = scroll.scrollHeight;
}

function setEmpty(empty) {
  $("app").classList.toggle("empty", empty);
}

// --- Sidebar -------------------------------------------------------------------------------

function setSidebar(open) {
  $("app").classList.toggle("sidebar-closed", !open);
  $("scrim").hidden = !(open && NARROW.matches);
}

function closeSidebarOnPhone() {
  if (NARROW.matches) setSidebar(false);
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

// --- Documents and the question's scope ----------------------------------------------------

async function loadDocuments() {
  const { data } = await api("/documents?limit=100");
  state.documents = data.items;
  for (const doc of data.items) {
    if (state.scope.has(doc.id)) state.scope.set(doc.id, { title: doc.title, status: doc.status });
  }
  $("doc-count").textContent = data.total ? String(data.total) : "";
  renderLibrary();
  renderChips();
}

function renderLibrary() {
  const list = $("library");
  list.replaceChildren();
  if (!state.documents.length) {
    list.append(el("li", { className: "empty", text: "No documents yet. Upload one to get started." }));
    return;
  }
  for (const doc of state.documents) {
    const box = el("input", { attrs: { type: "checkbox", "aria-label": `Search only in ${doc.title}` } });
    box.checked = state.scope.has(doc.id);
    box.disabled = doc.status !== "completed";
    box.addEventListener("change", () => {
      if (box.checked) state.scope.set(doc.id, { title: doc.title, status: doc.status });
      else state.scope.delete(doc.id);
      renderChips();
    });
    const status = doc.status === "completed" ? "" : doc.status;
    const label = el("label", {}, box, el("span", { className: "t", text: doc.title, attrs: { title: doc.filename } }));
    const badge = status ? el("span", { className: `s ${status}`, text: status }) : null;
    if (badge && doc.error_message) badge.title = doc.error_message;
    list.append(el("li", {}, label, badge));
  }
}

function renderChips() {
  const chips = $("chips");
  chips.replaceChildren();
  for (const [id, doc] of state.scope) {
    const status = { queued: "Waiting…", processing: "Processing…", failed: "Failed" }[doc.status];
    const remove = iconButton("close", `Remove ${doc.title}`, () => {
      state.scope.delete(id);
      renderChips();
      renderLibrary();
    });
    chips.append(
      el(
        "span",
        { className: "chip", attrs: { title: doc.title } },
        icon("files"),
        el("span", { className: "t", text: doc.title }),
        status ? el("span", { className: `s ${doc.status}`, text: status }) : null,
        remove,
      ),
    );
  }
  updateSend();
}

function scopeReady() {
  return [...state.scope.values()].every((doc) => doc.status === "completed");
}

async function upload(file) {
  const form = new FormData();
  form.append("file", file);
  const placeholder = `upload-${Date.now()}`;
  state.scope.set(placeholder, { title: file.name, status: "queued" });
  renderChips();
  try {
    const { data } = await api("/documents", { method: "POST", form });
    state.scope.delete(placeholder);
    state.scope.set(data.id, { title: data.title, status: data.status });
    renderChips();
    if (data.status !== "completed" && data.job_id) await waitForJob(data.job_id, data.id);
  } catch (error) {
    state.scope.delete(placeholder);
    renderChips();
    alert(`Upload failed: ${describeError(error)}`);
  } finally {
    await loadDocuments().catch(() => {});
  }
}

async function waitForJob(jobId, documentId) {
  const deadline = Date.now() + JOB_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, JOB_POLL_MS));
    const { data: job } = await api(`/jobs/${jobId}`);
    const doc = state.scope.get(documentId);
    if (doc) {
      doc.status = job.status === "retrying" ? "processing" : job.status;
      renderChips();
    }
    if (job.status === "completed" || job.status === "failed") return;
  }
}

// --- History -------------------------------------------------------------------------------

function dayBucket(date) {
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  const days = Math.floor((startOfToday - new Date(date).setHours(0, 0, 0, 0)) / 86400000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days <= 7) return "Previous 7 days";
  if (days <= 30) return "Previous 30 days";
  return "Older";
}

async function loadHistory() {
  const { data } = await api("/chats?limit=100");
  // Newest first; one entry per conversation, titled by its first question.
  const conversations = new Map();
  for (const chat of data.items) {
    const entry = conversations.get(chat.conversation_id) ?? { latest: chat.created_at };
    entry.title = chat.question; // ends as the oldest one
    conversations.set(chat.conversation_id, entry);
  }
  const history = $("history");
  history.replaceChildren();
  if (!conversations.size) history.append(el("p", { className: "none", text: "No chats yet." }));
  let bucket = null;
  for (const [conversationId, { title, latest }] of conversations) {
    const name = dayBucket(latest);
    if (name !== bucket) {
      history.append(el("h3", { text: name }));
      bucket = name;
    }
    const button = el("button", { text: title, attrs: { type: "button", title } });
    if (conversationId === state.conversationId) button.classList.add("active");
    button.addEventListener("click", () => showConversation(conversationId, button));
    history.append(button);
  }
}

async function showConversation(conversationId, button) {
  document.querySelectorAll(".history button.active").forEach((b) => b.classList.remove("active"));
  button?.classList.add("active");
  closeSidebarOnPhone();
  closeSources();
  $("thread").replaceChildren();
  setEmpty(false);
  state.conversationId = conversationId;
  try {
    const { data } = await api(`/chats?limit=100&conversation_id=${conversationId}`);
    const traces = await Promise.all(
      data.items.reverse().map((chat) => api(`/chats/${chat.id}`).then((r) => r.data)),
    );
    // Continue with the documents the last question was limited to.
    state.scope.clear();
    for (const id of traces.at(-1)?.document_ids ?? []) {
      const doc = state.documents.find((d) => d.id === id);
      if (doc) state.scope.set(id, { title: doc.title, status: doc.status });
    }
    renderChips();
    for (const chat of traces) {
      appendUser(chat.question);
      const content = appendAssistant();
      if (chat.error_code) renderError(content, `This question failed (${chat.error_code}).`);
      else showAnswer(content, { ...chat, usage: { prompt_tokens: chat.prompt_tokens, completion_tokens: chat.completion_tokens } });
    }
  } catch (error) {
    renderError(appendAssistant(), describeError(error));
  }
}

// --- Thread --------------------------------------------------------------------------------

function newChat() {
  state.conversationId = null;
  state.scope.clear();
  renderChips();
  closeSources();
  $("thread").replaceChildren();
  setEmpty(true);
  document.querySelectorAll(".history button.active").forEach((b) => b.classList.remove("active"));
  closeSidebarOnPhone();
  $("question").focus();
}

function appendUser(question) {
  $("thread").append(el("div", { className: "msg user" }, el("div", { className: "bubble", text: question })));
  scrollToEnd();
}

function appendAssistant() {
  const content = el("div", { className: "content" });
  $("thread").append(el("div", { className: "msg assistant" }, el("div", { className: "avatar", text: "◆" }), content));
  return content;
}

function renderThinking(content, label) {
  const node = el("div", { className: "thinking" }, el("span", { className: "pulse" }), el("span", { text: label }));
  content.append(node);
  scrollToEnd();
  return () => node.remove();
}

function renderError(content, message) {
  content.append(el("div", { className: "answer error", text: message }));
  scrollToEnd();
}

// One answer: what was searched, the text with citation buttons, source pills, actions.
function showAnswer(content, data) {
  if (data.search_query) {
    content.append(el("div", { className: "searched" }, icon("search"), el("span", { text: `Searched for “${data.search_query}”` })));
  }
  const answer = el("div", { className: "answer" });
  if (data.abstained) {
    answer.classList.add("abstained");
    answer.textContent = abstentionMessage(data.abstain_reason);
  } else {
    answer.innerHTML = renderAnswer(data.answer); // escaped by renderAnswer
  }
  answer.addEventListener("click", (event) => {
    const marker = event.target.closest(".cite")?.dataset.cite;
    if (marker) openSources(data, Number(marker));
  });
  content.append(answer);

  if (data.citations?.length) {
    const pills = el("div", { className: "pills" });
    for (const citation of data.citations) {
      const pill = el(
        "button",
        { className: "pill", attrs: { type: "button", title: locationLabel(citation) || citation.document_title } },
        el("span", { className: "n", text: String(citation.marker) }),
        el("span", { className: "t", text: citation.document_title }),
      );
      pill.addEventListener("click", () => openSources(data, citation.marker));
      pills.append(pill);
    }
    content.append(pills);
  }

  const actions = el("div", { className: "actions" });
  if (!data.abstained) {
    const copy = iconButton("copy", "Copy", async () => {
      await navigator.clipboard?.writeText(plainAnswer(data.answer)).catch(() => {});
      copy.replaceChildren(icon("check"));
      setTimeout(() => copy.replaceChildren(icon("copy")), 1500);
    });
    actions.append(copy);
  }
  if (data.retrieved?.length) actions.append(iconButton("book", "Sources", () => openSources(data)));
  const meta = [];
  if (data.timings_ms?.total != null) meta.push(formatSeconds(data.timings_ms.total));
  const tokens = (data.usage?.prompt_tokens ?? 0) + (data.usage?.completion_tokens ?? 0);
  if (tokens) meta.push(`${tokens.toLocaleString()} tokens`);
  if (meta.length) actions.append(el("span", { className: "meta", text: meta.join(" · ") }));
  content.append(actions);
  scrollToEnd();
}

// --- Sources panel -------------------------------------------------------------------------

function openSources(data, marker) {
  const body = $("sources-body");
  body.replaceChildren();
  const passages = new Map((data.retrieved ?? []).map((hit) => [hit.chunk_id, hit]));
  const cited = new Set();
  let focus = null;

  for (const citation of data.citations ?? []) {
    cited.add(citation.chunk_id);
    const text = passages.get(citation.chunk_id)?.text ?? citation.quote;
    const item = sourceItem(String(citation.marker), citation.document_title, locationLabel(citation), text, { formatted: true });
    if (citation.marker === marker) focus = item;
    body.append(item);
  }
  const others = (data.retrieved ?? []).filter((hit) => !cited.has(hit.chunk_id));
  if (others.length) {
    body.append(el("h3", { text: data.citations?.length ? "Also searched" : "Passages found" }));
    for (const hit of others) {
      const item = sourceItem(String(hit.rank), hit.document_title, locationLabel(hit), hit.text ? previewText(hit.text, 600) : "");
      item.querySelector(".head").append(el("span", { className: "score", text: hit.score.toFixed(2) }));
      body.append(item);
    }
  }
  $("sources").hidden = false;
  if (focus) {
    focus.scrollIntoView({ block: "start" });
    focus.classList.add("highlight");
  } else {
    body.scrollTop = 0;
  }
}

function sourceItem(number, title, where, text, { formatted = false } = {}) {
  const body = el("div", { className: formatted ? "text formatted" : "text" });
  if (formatted) body.innerHTML = renderPassage(text); // escaped by renderPassage
  else body.textContent = text;
  return el(
    "div",
    { className: "source" },
    el("div", { className: "head" }, el("span", { className: "cite", text: number }), el("span", { className: "title", text: title })),
    where ? el("div", { className: "where", text: where }) : null,
    text ? body : null,
  );
}

function closeSources() {
  $("sources").hidden = true;
}

// --- Asking --------------------------------------------------------------------------------

async function ask(question) {
  const body = { question };
  const ids = [...state.scope.keys()];
  if (ids.length) body.document_ids = ids;
  if (state.conversationId) body.conversation_id = state.conversationId;
  setEmpty(false);
  appendUser(question);
  const content = appendAssistant();

  if (!state.retrievalOnly) {
    const stop = renderThinking(content, state.conversationId ? "Reading the conversation and searching…" : "Searching your documents…");
    try {
      const { data } = await api("/chat", { method: "POST", json: body });
      stop();
      state.conversationId = data.conversation_id;
      showAnswer(content, data);
      loadHistory().catch(() => {});
      return;
    } catch (error) {
      stop();
      if (!(error instanceof ApiError && error.code === "chat_disabled")) {
        renderError(content, describeError(error));
        return;
      }
      setRetrievalOnly(); // this deployment has no language model: show the evidence instead
    }
  }

  const stop = renderThinking(content, "Searching your documents…");
  try {
    const { data } = await api("/retrieve", { method: "POST", json: body });
    stop();
    const result = { retrieved: data.hits, citations: [], timings_ms: data.timings_ms };
    content.append(
      el("div", {
        className: "answer abstained",
        text: data.hits.length
          ? "This demo runs without a language model, so here are the passages an answer would come from."
          : "Nothing in your documents matches this question.",
      }),
    );
    if (data.hits.length) {
      const pills = el("div", { className: "pills" });
      for (const hit of data.hits) {
        const pill = el("button", { className: "pill", attrs: { type: "button" } }, el("span", { className: "n", text: String(hit.rank) }), el("span", { className: "t", text: hit.document_title }));
        pill.addEventListener("click", () => openSources(result));
        pills.append(pill);
      }
      content.append(pills);
    }
    scrollToEnd();
  } catch (error) {
    stop();
    renderError(content, describeError(error));
  }
}

function setRetrievalOnly() {
  state.retrievalOnly = true;
  $("mode").hidden = false;
  $("question").placeholder = "Search your documents";
}

// --- Composer ------------------------------------------------------------------------------

function autosize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight}px`;
}

function updateSend() {
  const ready = scopeReady();
  $("send").disabled = state.busy || !$("question").value.trim() || !ready;
  $("send").title = ready ? "Send" : "Waiting for your file to be processed";
}

function setMenu(open) {
  $("plus-menu").hidden = !open;
  $("plus").setAttribute("aria-expanded", String(open));
}

function wire() {
  const question = $("question");
  $("composer").addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = question.value.trim();
    if (!text || state.busy || !scopeReady()) return;
    state.busy = true;
    question.value = "";
    autosize(question);
    updateSend();
    try {
      await ask(text);
    } finally {
      state.busy = false;
      updateSend();
      question.focus();
    }
  });
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  question.addEventListener("input", () => {
    autosize(question);
    updateSend();
  });

  $("plus").addEventListener("click", (event) => {
    event.stopPropagation();
    setMenu($("plus-menu").hidden);
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".plus")) setMenu(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setMenu(false);
      closeSources();
    }
  });
  $("menu-upload").addEventListener("click", () => {
    setMenu(false);
    $("upload").click();
  });
  $("menu-scope").addEventListener("click", () => {
    setMenu(false);
    $("library-dialog").showModal();
  });
  $("upload").addEventListener("change", (event) => {
    const file = event.target.files[0];
    event.target.value = "";
    if (file) upload(file);
  });

  $("new-chat").addEventListener("click", newChat);
  $("new-chat-top").addEventListener("click", newChat);
  $("open-library").addEventListener("click", () => {
    closeSidebarOnPhone();
    $("library-dialog").showModal();
  });
  $("library-upload").addEventListener("click", () => {
    $("library-dialog").close();
    $("upload").click();
  });
  $("settings").addEventListener("click", async () => {
    storeKey("");
    await askForKey();
    await refresh();
  });
  $("close-sidebar").addEventListener("click", () => setSidebar(false));
  $("open-sidebar").addEventListener("click", () => setSidebar(true));
  $("scrim").addEventListener("click", () => setSidebar(false));
  $("close-sources").addEventListener("click", closeSources);
  $("key-dialog").addEventListener("cancel", (event) => {
    if (!state.apiKey) event.preventDefault(); // the page needs a key
  });
}

async function refresh() {
  await Promise.all([loadDocuments(), loadHistory()]);
}

async function start() {
  hydrateIcons();
  wire();
  setSidebar(!NARROW.matches);
  NARROW.addEventListener("change", (event) => setSidebar(!event.matches));
  if (!state.apiKey) await askForKey();
  try {
    await refresh();
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) {
      await askForKey("Your saved key is no longer accepted.");
      await refresh();
    } else {
      setEmpty(false);
      renderError(appendAssistant(), describeError(error));
    }
  }
  $("question").focus();
}

start();
