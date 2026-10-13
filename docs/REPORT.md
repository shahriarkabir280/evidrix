# Evidrix: which retrieval changes pay off for a technical knowledge assistant?

## 1. Question

Which retrieval changes improve **evidence retrieval**, **answer quality** and **operational
cost** for a question-answering assistant over technical documentation and research papers?

Five controlled changes were tested, one variable at a time, against a dense-retrieval baseline:
keyword search fused with dense search (E2), cross-encoder reranking (E3), smaller chunks (E4),
and how many passages reach the model (E5). Settings were chosen on the dev split; each
configuration then ran **once** on the held-out test split.

## 2. Setup

| | |
|---|---|
| Corpus | 63 documents in two collections. *Documentation*: FastAPI tutorial 0.143.0 (51 Markdown pages, MIT) and four Python 3.12 HOWTO PDFs (argparse, logging, regex, sorting; PSF-2.0). *Papers*: 8 arXiv papers on retrieval and RAG evaluation (RAGAS, ARES, RAG failure points, Self-RAG, Sentence-BERT, E5, ColBERTv2, BEIR; CC BY / CC BY-SA). Every file is pinned by version and sha256 in `evals/corpus/manifest.json` |
| Chunking | structure-aware (headings, paragraphs, code blocks kept whole), 400-token target / 480 max, 64-token overlap (E4: 256 / 307 / 42); each chunk keeps its exact character range, pages and section path |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d, 512 tokens), query instruction on, section path prepended to each chunk |
| Keyword search | Postgres full-text search (`english` config, OR of the question's lexemes, `ts_rank_cd`) |
| Fusion | Reciprocal Rank Fusion, k = 60, 20 candidates per leg |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, 20 candidates → top k |
| Abstention gate | reranker presets: no model call when the best reranker logit is below −2.84 (calibrated on dev) |
| Answer model | `qwen3-vl:8b-instruct` in Ollama 0.35, temperature 0, 8,192-token context, JSON output, prompt `answer_v2` |
| Judge | `gemma3:12b` (a different model family), prompts `judge_correctness_v1` and `judge_faithfulness_v1` |
| Hardware | MacBook Air M4 (fanless), 16 GB; embedder and reranker on the Apple GPU (MPS) |

## 3. Benchmark

Two benchmarks, written from the source documents only:

| | Questions | Factual | Identifier | Multi-hop | Distractor | Table | Unanswerable | Dev / test |
|---|---|---|---|---|---|---|---|---|
| Documentation | 60 | 20 | 10 | 10 | 8 | — | 12 | 36 / 24 |
| Papers | 24 | 7 | 3 | 4 | 2 | 4 | 4 | 13 / 11 |

- **Labels are verbatim quotes** from the source, resolved to character ranges at evaluation
  time, so the same labels work for any chunking (E4). A retrieved chunk counts as evidence when
  it holds at least half of a labeled quote.
- **Splits** were stratified by type and fixed before any experiment. The test split was run
  only for this report (the runner refuses it without `--final`).
- **Metrics.** Retrieval: Recall@5, MRR@5, nDCG@5 (answerable questions). Answers: correctness
  (correct 1 / partial 0.5 / wrong 0, against a reference answer) and faithfulness (share of
  the answer's claims supported by the passages the model saw), both by the judge; citation
  precision; abstention on unanswerable questions; false abstentions on answerable ones. Every
  mean comes with a 95% bootstrap interval, and comparisons use paired bootstrap differences.
- **The judge was calibrated first**, on 16 hand-labeled cases: `gemma3:12b` agreed with all
  16 verdicts, `llama3.1:8b` with 14 (it penalized correct extra detail).
- **Caveat:** one annotator, who also built the system, wrote the questions and labels; there
  was no second-annotator review. Some paper answers appear in several places but were labeled
  at one, which undercounts retrieval on those questions.

## 4. Results

RESULTS

## 5. Error analysis

ERRORS

## 6. Limitations

- **Small benchmark.** 35 test questions (28 answerable) give wide intervals; only large
  differences can be told apart from noise. Pooling the two benchmarks narrows them but mixes
  two kinds of documents.
- **Single annotator, author-written questions.** The questions were written by someone who
  knew the system, which may favor phrasings it handles.
- **LLM judge.** Calibrated on 16 cases, not validated at scale; the error analysis found
  verdicts that were wrong (section 5). Faithfulness grading cannot catch a number taken from
  the wrong row of a table.
- **One domain, one model.** Technical documentation and RAG papers, one 8B answer model; the
  conclusions may not carry over to other domains or stronger models.
- **`ts_rank_cd` is not BM25**, so "hybrid" here means Postgres full-text search, not a tuned
  lexical retriever.
- **Latency on a laptop.** Generation time on the fanless M4 varied about 4× with thermal and
  memory state (docs/perf.md), so latency is reported but not compared between configurations;
  prompt tokens are the stable cost measure.

## 7. Future work

- OCR for scanned PDFs, and table extraction that keeps rows together.
- A BM25 keyword leg, and query rewriting for vocabulary mismatches.
- A larger benchmark with a second annotator and evidence labeled at every location.
- A dedicated vector database compared with pgvector at larger scale.

## 8. Reproduce

```bash
make up && make migrate && make corpus
uv run python -m app.cli ingest-dir evals/corpus/raw                       # index bge-small_c400_o64
uv run python -m app.cli index-version create bge-small_c256_o42 --chunk-size 256
uv run python -m app.cli index-version activate bge-small_c256_o42 && uv run python -m app.cli reindex
uv run python -m app.cli index-version activate bge-small_c400_o64
OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_MAX_LOADED_MODELS=1 ollama serve        # separate terminal
caffeinate -i scripts/run_ablations.sh                                      # ~2.5 h on an M4
uv run python -m evals.ablations                                            # the tables in section 4
uv run python -m evals.failures evals/results/<E3 run>_full.json --text     # section 5
```

Every results file records the git commit, models, prompt versions, index version and the
sha256 of the benchmark and corpus manifest.
