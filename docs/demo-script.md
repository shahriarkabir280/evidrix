# Demo video script (≈ 3 minutes)

Recorded locally, because the public demo is retrieval-only. Before recording:

```bash
make up && OLLAMA_CONTEXT_LENGTH=8192 ollama serve      # separate terminal
make dev                                                # API on :8000
make worker                                             # separate terminal
export EVIDRIX_API_KEY=evx_...                          # app.cli create-key --email dev@evidrix.local
AUTH="Authorization: Bearer $EVIDRIX_API_KEY"
curl -s -H "$AUTH" -H 'Content-Type: application/json' -d '{"question":"warm up"}' localhost:8000/chat >/dev/null  # load the model first
```

Use a PDF the corpus doesn't have (any short technical PDF with a text layer).

| # | Show | Command / action | Say |
|---|---|---|---|
| 1 | Swagger UI | open `localhost:8000/docs` | "An API for questions over your own documents, answered only from them, with citations." |
| 2 | Upload | `curl -s -H "$AUTH" -F "file=@guide.pdf" localhost:8000/documents` | "Upload returns 202 at once; a worker extracts, chunks and embeds in the background." |
| 3 | Job status | `curl -s -H "$AUTH" localhost:8000/jobs/<job_id>` (twice) | "queued → processing → completed; retries and crash recovery are built in." |
| 4 | Answerable question | `curl -s -H "$AUTH" -H 'Content-Type: application/json' -d '{"question":"<something the PDF answers>","document_ids":["<id>"]}' localhost:8000/chat \| jq '{answer, citations}'` | Point at the citation's `page_start` and `quote`: "every claim points at a page." |
| 5 | Unanswerable question | the same with a question the PDF can't answer | "`abstained: true` — it says it doesn't know instead of guessing." |
| 6 | Retrieval debug | `uv run python -m app.cli search "OAuth2PasswordRequestForm" --preset dense -k 3`, then `--preset hybrid_rerank` | "An identifier query: dense search ranks it loosely, keyword search finds the exact name, the reranker orders them." |
| 7 | History | `curl -s -H "$AUTH" localhost:8000/chats/<chat id> \| jq '.timings_ms, .retrieved[0]'` | "Every chat is stored with its scores and timings: generation is ~94% of the latency." |
| 8 | Results | README results table / `docs/REPORT.md` | "On a held-out test split, reranking was the change that helped: MRR +0.19, outside the confidence interval. Fusion alone didn't help, and five passages did as well as eight with two-thirds of the tokens." |

Keep the terminal font large, and cut the waiting time during generation.
