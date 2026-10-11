# ADR-0002: bge-small-en-v1.5 embeddings with chunks of at most 480 tokens

**Status:** accepted (2026-10-11)

## Context
The roadmap suggested `all-MiniLM-L6-v2` with 400–600-token chunks. MiniLM reads at most 256
word-pieces and silently ignores the rest. Measured in component 2B: with a 300-token passage
followed by an unrelated paragraph, MiniLM's embedding was identical (cosine 1.0000) with or
without the paragraph — the paragraph was invisible to search, with no error.

## Decision
- Embedding model: `BAAI/bge-small-en-v1.5` (384 dimensions, 512-token input).
- Chunks target 400 tokens with a hard maximum of 480, counted with the model's own tokenizer,
  leaving room for 2 special tokens and a short "title > section" header added at embedding time.
  If the header would push a chunk past the limit, the header is dropped, never the chunk text.
- Queries get bge's instruction ("Represent this sentence for searching relevant passages: ").
  The model's sentence-transformers config leaves the query prompt empty, so it is added
  explicitly rather than relying on `encode_query()`.
- Each index version records the embedding model; ingestion refuses an embedder that does not
  match the active index version.

## Consequences
- No chunk text is silently truncated (asserted on all 670 corpus chunks).
- An 8-question preview showed no measurable benefit from the query instruction
  (MRR 0.823 without vs 0.812 with); experiment E7 will measure it on the full benchmark.
- Similarity scores are model-specific (bge sits around 0.6–0.8, MiniLM much lower), so any
  abstention threshold must be calibrated for this model.
- Embedding all chunks takes 7.5 s on Apple MPS vs 19.8 s on CPU, with identical vectors.
