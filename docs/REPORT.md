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

Twelve runs: six configurations × two benchmarks, each once, on the test split (35 questions,
28 answerable, 7 unanswerable). Output of `uv run python -m evals.ablations`:

Split: test. Questions per benchmark: {'docs': 24, 'papers': 11}. Mean [95% bootstrap CI]. ΔE1 / ΔE3: paired difference against that run (E2, E3 vs E1; E4, E5 vs E3), **bold** when its interval excludes 0. Retrieval metrics cut off at 5, so E5b's are E3's.

#### Retrieval, pooled (answerable questions; Recall/MRR/nDCG at 5)

| | Config | recall | mrr | ndcg |
|---|---|---|---|---|
| E1 | dense, top 5 | 0.696 [0.52, 0.86] | 0.571 [0.40, 0.73] | 0.563 [0.41, 0.72] |
| E2 | hybrid (RRF), top 5 | 0.661 [0.48, 0.82]<br>ΔE1 -0.036 [-0.12, +0.04] | 0.487 [0.33, 0.65]<br>ΔE1 -0.084 [-0.22, +0.03] | 0.498 [0.35, 0.65]<br>ΔE1 -0.064 [-0.16, +0.02] |
| E3 | hybrid + rerank, top 5 | 0.821 [0.68, 0.93]<br>**ΔE1 +0.125 [+0.04, +0.25]** | 0.764 [0.62, 0.90]<br>**ΔE1 +0.193 [+0.05, +0.35]** | 0.708 [0.58, 0.84]<br>**ΔE1 +0.145 [+0.04, +0.27]** |
| E4 | E3 with 256-token chunks | 0.768 [0.62, 0.88]<br>ΔE3 -0.054 [-0.18, +0.05] | 0.717 [0.59, 0.84]<br>ΔE3 -0.047 [-0.15, +0.07] | 0.676 [0.55, 0.80]<br>ΔE3 -0.031 [-0.12, +0.06] |
| E5a | E3 with top 3 | 0.732 [0.59, 0.88]<br>ΔE3 -0.089 [-0.20, +0.00] | 0.750 [0.61, 0.89]<br>ΔE3 -0.014 [-0.04, +0.00] | 0.671 [0.53, 0.82]<br>ΔE3 -0.037 [-0.08, +0.00] |
| E5b | E3 with top 8 | 0.821 [0.68, 0.93]<br>ΔE3 +0.000 [+0.00, +0.00] | 0.764 [0.62, 0.90]<br>ΔE3 +0.000 [+0.00, +0.00] | 0.708 [0.58, 0.84]<br>ΔE3 +0.000 [+0.00, +0.00] |

#### Answers, pooled (judge: gemma3:12b)

| | Config | correctness | faithfulness |
|---|---|---|---|
| E1 | dense, top 5 | 0.768 [0.61, 0.91] | 0.996 [0.99, 1.00] |
| E2 | hybrid (RRF), top 5 | 0.750 [0.59, 0.89]<br>ΔE1 -0.018 [-0.16, +0.12] | 0.917 [0.79, 1.00]<br>ΔE1 -0.083 [-0.21, +0.00] |
| E3 | hybrid + rerank, top 5 | 0.857 [0.75, 0.96]<br>ΔE1 +0.089 [-0.02, +0.21] | 1.000 [1.00, 1.00]<br>ΔE1 +0.004 [+0.00, +0.01] |
| E4 | E3 with 256-token chunks | 0.786 [0.64, 0.91]<br>ΔE3 -0.071 [-0.20, +0.02] | 0.990 [0.97, 1.00]<br>ΔE3 -0.010 [-0.03, +0.00] |
| E5a | E3 with top 3 | 0.821 [0.68, 0.95]<br>ΔE3 -0.036 [-0.12, +0.04] | 0.968 [0.92, 1.00]<br>ΔE3 -0.032 [-0.08, +0.00] |
| E5b | E3 with top 8 | 0.839 [0.71, 0.95]<br>ΔE3 -0.018 [-0.14, +0.11] | 1.000 [1.00, 1.00]<br>ΔE3 +0.000 [+0.00, +0.00] |

#### Abstention and operations, pooled

| | Config | Unanswerable caught | False abstentions | Citation precision | Prompt tokens / query | generate p50 | Errors |
|---|---|---|---|---|---|---|---|
| E1 | dense, top 5 | 7/7 | 3/28 | 0.64 | 1810 | 18.3 s | 0 |
| E2 | hybrid (RRF), top 5 | 7/7 | 4/28 | 0.639 | 1944 | 18.3 s | 0 |
| E3 | hybrid + rerank, top 5 | 7/7 | 1/28 | 0.698 | 1874 | 18.1 s | 0 |
| E4 | E3 with 256-token chunks | 7/7 | 3/28 | 0.72 | 1379 | 13.4 s | 0 |
| E5a | E3 with top 3 | 7/7 | 2/28 | 0.692 | 1269 | 12.4 s | 0 |
| E5b | E3 with top 8 | 7/7 | 2/28 | 0.667 | 2901 | 30.3 s | 1 |

#### Per benchmark: docs

| | Config | recall | mrr | correctness |
|---|---|---|---|---|
| E1 | dense, top 5 | 0.816 [0.63, 0.97] | 0.671 [0.47, 0.85] | 0.842 [0.68, 1.00] |
| E2 | hybrid (RRF), top 5 | 0.763 [0.58, 0.95]<br>ΔE1 -0.053 [-0.16, +0.00] | 0.559 [0.37, 0.75]<br>ΔE1 -0.112 [-0.27, +0.04] | 0.789 [0.58, 0.95]<br>ΔE1 -0.053 [-0.21, +0.11] |
| E3 | hybrid + rerank, top 5 | 0.868 [0.74, 0.97]<br>ΔE1 +0.053 [+0.00, +0.13] | 0.879 [0.72, 1.00]<br>**ΔE1 +0.208 [+0.03, +0.40]** | 0.947 [0.84, 1.00]<br>ΔE1 +0.105 [+0.00, +0.26] |
| E4 | E3 with 256-token chunks | 0.842 [0.68, 0.95]<br>ΔE3 -0.026 [-0.08, +0.00] | 0.807 [0.67, 0.94]<br>ΔE3 -0.072 [-0.20, +0.06] | 0.842 [0.68, 1.00]<br>ΔE3 -0.105 [-0.26, +0.00] |
| E5a | E3 with top 3 | 0.789 [0.63, 0.92]<br>ΔE3 -0.079 [-0.21, +0.00] | 0.868 [0.71, 1.00]<br>ΔE3 -0.011 [-0.03, +0.00] | 0.895 [0.79, 1.00]<br>ΔE3 -0.053 [-0.16, +0.00] |
| E5b | E3 with top 8 | 0.868 [0.74, 0.97]<br>ΔE3 +0.000 [+0.00, +0.00] | 0.879 [0.72, 1.00]<br>ΔE3 +0.000 [+0.00, +0.00] | 0.895 [0.74, 1.00]<br>ΔE3 -0.053 [-0.21, +0.11] |

#### Per benchmark: papers

| | Config | recall | mrr | correctness |
|---|---|---|---|---|
| E1 | dense, top 5 | 0.444 [0.11, 0.78] | 0.361 [0.11, 0.69] | 0.611 [0.33, 0.83] |
| E2 | hybrid (RRF), top 5 | 0.444 [0.17, 0.72]<br>ΔE1 +0.000 [-0.17, +0.17] | 0.337 [0.10, 0.63]<br>ΔE1 -0.024 [-0.22, +0.10] | 0.667 [0.44, 0.89]<br>ΔE1 +0.056 [-0.11, +0.22] |
| E3 | hybrid + rerank, top 5 | 0.722 [0.44, 1.00]<br>ΔE1 +0.278 [+0.00, +0.56] | 0.522 [0.29, 0.78]<br>ΔE1 +0.161 [-0.08, +0.42] | 0.667 [0.44, 0.89]<br>ΔE1 +0.056 [-0.17, +0.33] |
| E4 | E3 with 256-token chunks | 0.611 [0.39, 0.83]<br>ΔE3 -0.111 [-0.44, +0.22] | 0.528 [0.28, 0.78]<br>ΔE3 +0.006 [-0.17, +0.19] | 0.667 [0.44, 0.89]<br>ΔE3 +0.000 [-0.17, +0.17] |
| E5a | E3 with top 3 | 0.611 [0.33, 0.89]<br>ΔE3 -0.111 [-0.33, +0.00] | 0.500 [0.28, 0.78]<br>ΔE3 -0.022 [-0.07, +0.00] | 0.667 [0.39, 0.89]<br>ΔE3 +0.000 [-0.17, +0.17] |
| E5b | E3 with top 8 | 0.722 [0.44, 1.00]<br>ΔE3 +0.000 [+0.00, +0.00] | 0.522 [0.29, 0.78]<br>ΔE3 +0.000 [+0.00, +0.00] | 0.722 [0.50, 0.94]<br>ΔE3 +0.056 [+0.00, +0.17] |

**What the data supports**

- **Reranking is the change that helps (E3).** Against dense retrieval it raises Recall@5 by
  +0.125, MRR@5 by +0.193 and nDCG@5 by +0.145, each with an interval that excludes zero; the same
  result as on dev (MRR +0.155). Answer correctness moves the same way (+0.089, 0.768 → 0.857)
  and false abstentions drop from 3 to 1 of 28, but with 28 answerable questions the
  correctness interval still includes zero.
- **Hybrid search alone does not help (E2).** Fusing Postgres keyword search with dense results
  lowers every retrieval metric slightly (MRR −0.084) and adds a false abstention. The keyword
  leg is only useful once a reranker sorts its candidates. This matches dev.
- **Smaller chunks don't pay off (E4).** 256-token chunks cut prompt tokens by 26% (1,874 →
  1,379) but lose on every quality measure (correctness −0.071, Recall −0.054; neither
  significant) and triple the false abstentions (1 → 3). Questions whose evidence spans a
  paragraph or two fit one 400-token chunk better.
- **Five passages is the right depth (E5).** Three passages save 32% of the prompt tokens with
  a small, non-significant loss (correctness −0.036); eight cost 55% more tokens (2,901) and
  ~70% more generation time for no gain (−0.018), and the longest prompts broke one judge reply.
  Retrieval metrics are cut off at 5, so E5b's equal E3's by construction; E5a's are lower
  because only three passages are retrieved.
- **Abstention is solid in every configuration:** all 7 unanswerable questions abstained
  everywhere. Faithfulness is at or near 1.0 throughout, so the answer model rarely says things
  its sources don't; the losses are in finding the evidence and in answering completely.
- **Papers are harder than documentation.** With E3, Recall@5 is 0.87 on docs and 0.72 on
  papers, correctness 0.95 vs 0.67. Paper questions often need two documents or a table.

**Cost per configuration.** Every answer costs $0 with the local model, so prompt tokens are the
cost measure: E3 sends ~1,870 tokens per question. Generation times (12–30 s) follow the prompt
length but come from a laptop whose speed varied with its thermal state, so they are indicative
only (docs/perf.md).

**Corrections.** The error analysis found a bug: `x[1]` in a code answer was parsed as a
citation marker, and the judge, which strips markers, graded `lambda x: x`. After the fix
(commit b3de835) the affected question (q037, in E1, E3 and E5b) was re-scored with
`evals/rescore.py` (incorrect → correct); each results file records this in its provenance.
One judge reply in E5b (papers, p009) was cut off, so that row has no faithfulness score.


## 5. Error analysis

All failures of the final configuration (E3) were classified by reading the question, its
labeled evidence, the retrieved passages, the answer and the judge's reasons
(`evals/failures.py`): 6 test failures (of 35) and 8 dev failures (of 49).

| Category | Test | Dev | Total |
|---|---|---|---|
| **Retrieval:** one hop of a multi-hop question missing | 2 | 2 | 4 |
| **Retrieval:** evidence not in the top 5 (another passage ranked above it) | 1 | 1 | 2 |
| **Extraction:** a table's rows lost in PDF text extraction | 1 | 1 | 2 |
| **Generation:** false abstention although the evidence was retrieved | 1 | 1 | 2 |
| **Judge:** correct extra detail graded "partial" | 1 | 2 | 3 |
| **Judge:** a verbatim-supported claim graded unsupported | 0 | 1 | 1 |
| Missed abstention / followed an injected instruction | 0 | 0 | 0 |

So 6 of 14 are retrieval failures, 2 extraction, 2 generation, and 4 are the judge's: measured
correctness understates the system somewhat.

**Examples**

- *One hop missing (p018, test).* "Ragas and ARES both evaluate RAG systems. Which of them needs
  human-annotated data, and how much?" All five retrieved passages come from the ARES paper
  (reranker scores 4.8–5.3); the Ragas sentence "without having to rely on ground truth human
  annotations" is not among them. The model then guesses that Ragas also uses human annotations.
  In p019 the same thing happens across E5 and BEIR: without the BEIR passage, the model takes
  "15 datasets" from the E5 paper (which evaluates on 15 of BEIR's 18).
- *Table rows lost (p014, test).* Asked for the best system on the STS benchmark, the model
  answers "SRoBERTa-NLI-STSb-large … 88.77": the score is right, the row is wrong. The extracted
  table reads column by column, so a number is no longer next to its model name. The judge's
  faithfulness check passes it, because 88.77 is in the source.
- *False abstention with perfect retrieval (q022, test).* "Which fields does
  OAuth2PasswordRequestForm declare?" — the right section is ranked first (reranker 5.22), yet
  the answer is an abstention with reason `no_valid_citations`: the model's citations did not
  survive validation, and the system refuses rather than show an uncited answer.
- *Judge strictness (p004, test).* The answer says Self-RAG's reflection tokens are inserted by
  a trained critic model, which the paper states; the reference answer doesn't mention it, so
  the judge called the answer partial.

**What to work on next.** Multi-hop retrieval: the reranker ranks passages one at a time, so one
document's five strong passages crowd out the second document a question needs. Limiting
passages per document or decomposing the question into one query per hop targets the largest
category. Then table-aware PDF extraction, and logging the model's raw reply on a
`no_valid_citations` abstention to see why its citations fail validation.


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
