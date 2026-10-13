// node --test tests/ui
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  abstentionMessage,
  citationNumbers,
  escapeHtml,
  formatSeconds,
  locationLabel,
  previewText,
  renderAnswer,
} from "../../app/ui/render.js";

test("escapes everything that could be markup", () => {
  assert.equal(escapeHtml(`<img src=x onerror="alert('x')">&`), "&lt;img src=x onerror=&quot;alert(&#39;x&#39;)&quot;&gt;&amp;");
});

test("an injected script in an answer is shown as text, never as HTML", () => {
  const html = renderAnswer('See [1]. <script>steal()</script> <a href="http://evil.example">login</a>');

  assert.ok(!html.includes("<script>"));
  assert.ok(!html.includes("<a "));
  assert.ok(html.includes("&lt;script&gt;"));
});

test("citation markers become source buttons, runs and lists included", () => {
  const html = renderAnswer("Use Cookie [1]. See also [2][3] and [4, 5].");

  for (const n of [1, 2, 3, 4, 5]) assert.ok(html.includes(`data-cite="${n}"`), `marker ${n}`);
  assert.ok(!html.includes("[1]"));
});

test("indexing in code is not a citation, inside or outside backticks", () => {
  const html = renderAnswer("Sort with key=lambda x: x[1] or `rows[2]`, as shown [3].");

  assert.ok(html.includes("x[1]"));
  assert.ok(html.includes("<code>rows[2]</code>"));
  assert.ok(!html.includes('data-cite="1"'));
  assert.ok(!html.includes('data-cite="2"'));
  assert.ok(html.includes('data-cite="3"'));
});

test("fenced code blocks are escaped and keep their language", () => {
  const html = renderAnswer("Like this:\n```python\nif a < b:\n    print(x[1])\n```\nDone [1].");

  assert.ok(html.includes('<pre data-language="python"><code>if a &lt; b:\n    print(x[1])</code></pre>'));
  assert.ok(html.includes('data-cite="1"'));
});

test("a language name cannot break out of its attribute", () => {
  const html = renderAnswer('```" onmouseover="x\ncode\n```');

  assert.ok(html.includes('data-language="&quot; onmouseover=&quot;x"'));
});

test("citation numbers come in order of first appearance, once each", () => {
  assert.deepEqual(citationNumbers("A [2]. B [1][2]. C x[9]. D [3, 1]."), [2, 1, 3]);
});

test("locations show the section and the page or page range", () => {
  assert.equal(locationLabel({ section_path: "Cookies > Import", page_start: null }), "Cookies > Import");
  assert.equal(locationLabel({ section_path: null, page_start: 3, page_end: 3 }), "p. 3");
  assert.equal(locationLabel({ section_path: "Intro", page_start: 3, page_end: 5 }), "Intro · pp. 3–5");
});

test("durations read naturally", () => {
  assert.equal(formatSeconds(412.4), "412 ms");
  assert.equal(formatSeconds(18123), "18.1 s");
  assert.equal(formatSeconds(null), "");
});

test("abstentions explain themselves", () => {
  assert.match(abstentionMessage("no_sources"), /Upload/);
  assert.match(abstentionMessage("model"), /couldn't find/);
});

test("previews drop Markdown markers and stay short", () => {
  const text = "## Declare `Cookie` parameters\n\nThen:\n```python\nfrom fastapi import Cookie\n```";

  assert.equal(previewText(text), "Declare `Cookie` parameters Then: from fastapi import Cookie");
  assert.equal(previewText("a".repeat(300), 10), "aaaaaaaaa…");
});
