#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Generalization check: does the length-equalized lexical+LR signal from
# Experiment 2 hold up under a different attack strategy (TAP) on the same
# target model (llama-3.1-8b-instruct, the only model TAP data exists for)?
# 10-seed sweep, matching Experiment 2's methodology.

SEEDS=(0 1 2 3 4 5 6 7 8 9)

for seed in "${SEEDS[@]}"; do
    analysis \
        --run_name "analysis/ablation_crescendo/seed${seed}" \
        --seed "${seed}" \
        --features.trim_to_first_n_turns 6 \
        --features.use_embeddings "lexical" \
        --features.use_roles "conversation" "assistant" "user" \
        --features.use_features "l2norm" "catch22" \
        --classifiers.logistic_regression.C 1.0 \
        --dataset_dirs "experiments/crescendo/llama-3.1-8b-instruct"
done

for seed in "${SEEDS[@]}"; do
    analysis \
        --run_name "analysis/ablation_tap/seed${seed}" \
        --seed "${seed}" \
        --features.trim_to_first_n_turns 6 \
        --features.use_embeddings "lexical" \
        --features.use_roles "conversation" "assistant" "user" \
        --features.use_features "l2norm" "catch22" \
        --classifiers.logistic_regression.C 1.0 \
        --dataset_dirs "experiments/tap/llama-3.1-8b-instruct"
done
