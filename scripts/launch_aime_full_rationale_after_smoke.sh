#!/usr/bin/env bash
set -euo pipefail

release_root="${RELEASE_ROOT:-/root/autodl-tmp/topology_MAS}"
smoke_dir="${SMOKE_DIR:-/root/autodl-tmp/aime_full_rationale_smoke_complete_p15_v1}"
smoke_pid_file="${SMOKE_PID_FILE:-/root/autodl-tmp/aime_full_rationale_smoke_complete_p15_v1.pid}"
main_dir="${MAIN_DIR:-/root/autodl-tmp/aime_original_2026_qwen3_4b_clean_mas_full_rationale_n5_h3_v1}"
main_log="${MAIN_LOG:-/root/autodl-tmp/aime_original_2026_qwen3_4b_clean_mas_full_rationale_n5_h3_v1.log}"
workers="${MAX_WORKERS:-12}"

smoke_pid="$(cat "$smoke_pid_file")"
while kill -0 "$smoke_pid" 2>/dev/null; do
  sleep 30
done

if [[ ! -s "$smoke_dir/summary.json" ]]; then
  echo "smoke failed; refusing to launch main batch" >&2
  tail -100 /root/autodl-tmp/aime_full_rationale_smoke_complete_p15_v1.log >&2 || true
  exit 2
fi

cd "$release_root"
export PYTHONPATH=src
/root/miniconda3/bin/python scripts/audit_aime_full_rationale_batch.py \
  --batch-dir "$smoke_dir" \
  --output-dir "$smoke_dir/audit"

if [[ -e "$main_dir/manifest.json" ]]; then
  echo "main output already initialized; refusing ambiguous relaunch" >&2
  exit 3
fi

OUTPUT_DIR="$main_dir" MAX_WORKERS="$workers" \
  bash scripts/run_aime_full_rationale_clean_mas.sh > "$main_log" 2>&1
