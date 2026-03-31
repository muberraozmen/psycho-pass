#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

config_file="base.yaml"
run_name="example"
max_concurrency=10

dataset_names=(
  "airt_fairness"
)

num_samples=2

attack_types=(
  "crescendo"
)

attack_max_turns=4

adversarial_model_names=(
  "meta-llama/llama-3.1-8b-instruct"
)

scoring_model_names=(
  "openai/gpt-oss-120b"
)

objective_model_names=(
  "meta-llama/llama-3.1-8b-instruct"
)

for dataset_name in "${dataset_names[@]}"; do
  for attack_type in "${attack_types[@]}"; do
    for adversarial_model_name in "${adversarial_model_names[@]}"; do
      for scoring_model_name in "${scoring_model_names[@]}"; do
        for objective_model_name in "${objective_model_names[@]}"; do
          uv run python run.py \
            --config_file "$config_file" \
            --run_name "$run_name" \
            --max_concurrency "$max_concurrency" \
            --seeds.dataset_name "$dataset_name" \
            --seeds.num_samples "$num_samples" \
            --attack.type "$attack_type" \
            --attack.max_turns "$attack_max_turns" \
            --attack.adversarial.model_name "$adversarial_model_name" \
            --attack.scoring.model_name "$scoring_model_name" \
            --attack.objective.model_name "$objective_model_name"
        done
      done
    done
  done
done