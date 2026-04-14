#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

config_file="base.yaml"
run_name="objective_model_comparisons"
max_concurrency=32

dataset_names = (
    "adv_bench"
    "harmbench"
    "airt_fairness"
    "airt_harassment"
    "airt_hate"
    "airt_illegal"
    "airt_imminent_crisis"
    "airt_leakage"
    "airt_malware"
    "airt_misinformation"
    "airt_scams"
    "airt_sexual"
    "airt_violence"
)

num_samples=1

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
    "openai/gpt-oss-120b"
    "mistralai/mistral-small-3.2-24b-instruct"
    "qwen/qwen-2.5-7b-instruct"
    "google/gemma-3-27b-it"
    "deepseek/deepseek-v3.2"
)


for dataset_name in "${dataset_names[@]}"; do
  for attack_type in "${attack_types[@]}"; do
    for adversarial_model_name in "${adversarial_model_names[@]}"; do
      for scoring_model_name in "${scoring_model_names[@]}"; do
        for objective_model_name in "${objective_model_names[@]}"; do
          generation \
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