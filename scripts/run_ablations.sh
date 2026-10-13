#!/usr/bin/env bash
# The Part 8 ablations, each run once on the held-out test split (docs/REPORT.md).
#
#   caffeinate -i scripts/run_ablations.sh        # macOS: keep the machine awake (~2.5 h)
#
# Needs the corpus ingested into both index versions (bge-small_c400_o64, the default, and
# bge-small_c256_o42 for E4) and Ollama running with qwen3-vl:8b-instruct and gemma3:12b.
set -euo pipefail

run() {  # benchmark preset
  echo "=== $1 $2 $(date +%H:%M:%S)"
  uv run python -m evals.run --benchmark "$1" --preset "$2" --split test --mode full --final \
    2>&1 | grep -E "^Wrote|Error|Traceback" || true
}

for benchmark in docs papers; do
  for preset in dense hybrid hybrid_rerank hybrid_rerank_k3 hybrid_rerank_k8; do  # E1 E2 E3 E5
    run "$benchmark" "$preset"
  done
done

# E4: the same pipeline as E3 over 256-token chunks.
uv run python -m app.cli index-version activate bge-small_c256_o42
trap 'uv run python -m app.cli index-version activate bge-small_c400_o64' EXIT
for benchmark in docs papers; do
  run "$benchmark" hybrid_rerank
done
echo "=== all done $(date +%H:%M:%S)"
