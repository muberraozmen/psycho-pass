#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

shopt -s nullglob
configs=(configs/model_iterations/*.yaml)
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No YAML configs found in configs/model_iterations/" >&2
  exit 1
fi

for path in "${configs[@]}"; do
  stem=$(basename "$path" .yaml)
  echo "=== model_iterations/${stem}.yaml ==="
  uv run python run.py \
    --config_file "model_iterations/${stem}.yaml" \
    --run_name "model_iterations_${stem}"
done
