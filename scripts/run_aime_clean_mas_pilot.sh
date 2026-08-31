#!/usr/bin/env bash
set -euo pipefail

release_root="${RELEASE_ROOT:-$(pwd)}"
output_dir="${OUTPUT_DIR:?set OUTPUT_DIR to a new experiment directory}"
base_url="${BASE_URL:-http://127.0.0.1:8000/v1}"
model="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
workers="${MAX_WORKERS:-64}"

cd "$release_root"
export PYTHONPATH=src

/root/miniconda3/bin/python -m topology_mas.execution.batch_cli \
  --tasks data/aime/original_2026.jsonl \
  --task-format aime-free-response \
  --graphs data/aime/clean_mas_n5_h3_v1/graphs.jsonl \
  --output-dir "$output_dir" \
  --independent-round-zero \
  --clean-only \
  --experiment-seeds 0 \
  --assignment-seeds 0 \
  --model "$model" \
  --expected-returned-model "$model" \
  --base-url "$base_url" \
  --no-auth \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --max-output-tokens 1024 \
  --aime-private-max-output-tokens 16384 \
  --aime-summary-temperature 0.0 \
  --message-order-seed 0 \
  --horizon-policy fixed \
  --timeout-seconds 1800 \
  --max-attempts 3 \
  --max-workers "$workers"
