// Turns API text into safe HTML for the chat UI. Pure functions, tested in tests/ui.
//
// Everything shown comes from documents or from a model that read them, so all of it is
// untrusted: text is escaped before any markup is added, and the only markup added is code
// formatting and citation buttons. No links, no images, no raw HTML (docs/security.md).

// The same rule as CITATION_RUN in app/rag/generation.py: a run of [n] markers that is not
// glued to a word, ")" or "]" (so x[1] in code stays code).
const CITATION_RUN = /(?<![\w)\]])((?:\[\d+(?:\s*,\s*\d+)*\])+)/g;

export function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/** The citation numbers in an answer, in order of first appearance. */
export function citationNumbers(text) {
  const seen = [];
  for (const run of String(text).matchAll(CITATION_RUN)) {
    for (const n of run[1].match(/\d+/g)) {
      const number = Number(n);
      if (!seen.includes(number)) seen.push(number);
    }
  }
  return seen;
}

function citationButtons(run) {
  return run
    .match(/\d+/g)
    .map((n) => `<button type="button" class="cite" data-cite="${n}" aria-label="Source ${n}">${n}</button>`)
    .join("");
}

function renderProse(text) {
  // Inline code first, so brackets inside `code` are never turned into citations.
  return text
    .split(/(`[^`\n]+`)/g)
    .map((part) => {
      if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
        return `<code>${escapeHtml(part.slice(1, -1))}</code>`;
      }
      return escapeHtml(part)
        .replace(CITATION_RUN, (_, run) => citationButtons(run))
        .replaceAll("\n", "<br>");
    })
    .join("");
}

/** An answer as HTML: fenced code blocks, inline code, citation buttons, line breaks. */
export function renderAnswer(text) {
  return String(text ?? "")
    .split(/(```[^\n]*\n[\s\S]*?```)/g)
    .map((part) => {
      const block = part.match(/^```([^\n]*)\n([\s\S]*?)```$/);
      if (block) {
        const language = escapeHtml(block[1].trim());
        return `<pre${language ? ` data-language="${language}"` : ""}><code>${escapeHtml(block[2].replace(/\n$/, ""))}</code></pre>`;
      }
      return renderProse(part);
    })
    .join("");
}

/** A passage as one line of plain text for a preview: no Markdown heading or fence markers. */
export function previewText(text, limit = 280) {
  const plain = String(text ?? "")
    .replace(/```[^\n]*\n?/g, " ")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\s+/g, " ")
    .trim();
  return plain.length > limit ? `${plain.slice(0, limit - 1)}…` : plain;
}

/** "Cookie Parameters > Import Cookie · p. 3–4", for a source location from the API. */
export function locationLabel(source) {
  const parts = [];
  if (source.section_path) parts.push(source.section_path);
  if (source.page_start != null) {
    const same = source.page_end == null || source.page_end === source.page_start;
    parts.push(same ? `p. ${source.page_start}` : `pp. ${source.page_start}–${source.page_end}`);
  }
  return parts.join(" · ");
}

export function formatSeconds(milliseconds) {
  if (milliseconds == null) return "";
  return milliseconds < 1000 ? `${Math.round(milliseconds)} ms` : `${(milliseconds / 1000).toFixed(1)} s`;
}

/** What to say when the system abstained, by the API's abstain_reason. */
export function abstentionMessage(reason) {
  if (reason === "no_sources") {
    return "There are no processed documents to search yet. Upload a file first.";
  }
  return "I couldn't find this in your documents, so I won't guess.";
}
