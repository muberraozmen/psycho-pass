#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

dataset_dirs=(
    "experiments/crescendo/gpt-oss-120b"
    "experiments/crescendo/llama-3.1-8b-instruct"
    "experiments/crescendo/mistral-small-3.2-24b-instruct"
    "experiments/crescendo/qwen-2.5-7b-instruct"
)

analysis \
    --run_name "analysis/experiment1a" \
    --features.use_embeddings "semantic" \
    --features.use_roles "conversation" \
    --features.use_features "l2norm" "catch22" "executed_turns" \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"

analysis \
    --run_name "analysis/experiment1b" \
    --features.use_embeddings "semantic" \
    --features.use_roles "conversation" \
    --features.use_features "l2norm" "catch22" \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"

analysis \
    --run_name "analysis/experiment1c" \
    --features.use_embeddings "lexical" \
    --features.use_roles "conversation" \
    --features.use_features "l2norm" "catch22" "executed_turns" \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"

analysis \
    --run_name "analysis/experiment1d" \
    --features.use_embeddings "lexical" \
    --features.use_roles "conversation" \
    --features.use_features "l2norm" "catch22" \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"

