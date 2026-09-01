#!/usr/bin/env bash
set -euo pipefail

release_root="${RELEASE_ROOT:-/root/autodl-tmp/topology_MAS}"
batch_dir="${BATCH_DIR:-/root/autodl-tmp/aime_original_2026_qwen3_4b_clean_mas_full_rationale_n5_h3_v1}"
batch_pid="${BATCH_PID:?set BATCH_PID to the active batch process}"
reference_csv="${ROUND_ZERO_REFERENCE:-/root/autodl-tmp/aime_original_2026_qwen3_4b_round0_16k_formal_v1_analysis/per_task_solve_rates.csv}"
analysis_dir="${ANALYSIS_DIR:-/root/autodl-tmp/aime_original_2026_qwen3_4b_clean_mas_full_rationale_n5_h3_v1_analysis}"

while kill -0 "$batch_pid" 2>/dev/null; do
  sleep 60
done

if [[ ! -s "$batch_dir/summary.json" ]]; then
  echo "main batch ended without a complete summary; refusing analysis" >&2
  exit 2
fi

cd "$release_root"
export PYTHONPATH=src
/root/miniconda3/bin/python scripts/audit_aime_full_rationale_batch.py \
  --batch-dir "$batch_dir" \
  --output-dir "$analysis_dir/audit"
/root/miniconda3/bin/python scripts/analyze_aime_clean_mas.py \
  --batch-dir "$batch_dir" \
  --batch-label full_rationale \
  --communication-protocol full-rationale \
  --round-zero-reference "$reference_csv" \
  --output-dir "$analysis_dir" \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260901
