---
title: Evidrix
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Evidence-grounded retrieval over technical docs and papers
---

# Evidrix — retrieval demo

This Space runs the [Evidrix](https://github.com/shahriarkabir280/evidrix) API in
**retrieval-only** mode: `POST /retrieve` returns the ranked passages (dense + keyword search,
fused and reranked) for a question, with per-stage scores. Answer generation (`/chat`) needs an
8B model and runs in the project's local setup (see its README).

Open `/docs` for the interactive API. Requests need an API key (`Authorization: Bearer evx_…`).
