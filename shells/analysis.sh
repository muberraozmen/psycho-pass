#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

dataset_dirs=(
    "experiments/generation/gpt-oss-120b"
    "experiments/generation/llama-3.1-8b-instruct"
    "experiments/generation/mistral-small-3.2-24b-instruct"
    "experiments/generation/qwen-2.5-7b-instruct"
)

analysis \
    --run_name "analysis/exp1a" \
    --features.use_embeddings "semantic" "lexical" \
    --features.use_roles "conversation" "assistant" "user" \
    --features.use_features "l2norm" "catch22" "executed_turns" \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"


analysis \
    --run_name "analysis/exp1b" \
    --features.use_embeddings "semantic" "lexical" \
    --features.use_roles "conversation" "assistant" "user" \
    --features.use_features "l2norm" "catch22" \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"


analysis \
    --run_name "analysis/exp2a" \
    --features.min_executed_turns 4 \
    --features.use_embeddings "semantic" \
    --features.use_roles "conversation" "assistant" "user" \
    --features.use_features "l2norm" "catch22" "executed_turns" \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"


analysis \
    --run_name "analysis/exp2b" \
    --features.min_executed_turns 4 \
    --features.use_embeddings "lexical" \
    --features.use_roles "conversation" "assistant" "user" \
    --features.use_features "l2norm" "catch22" "executed_turns" \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"


analysis \
    --run_name "analysis/exp3a" \
    --features.min_executed_turns 6 \
    --features.use_embeddings "semantic" \
    --features.use_roles "conversation" "assistant" "user" \
    --features.use_features "l2norm" "catch22" \
    --features.trim_to_first_n_turns 6 \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"


analysis \
    --run_name "analysis/exp3b" \
    --features.min_executed_turns 6 \
    --features.use_embeddings "lexical" \
    --features.use_roles "conversation" "assistant" "user" \
    --features.use_features "l2norm" "catch22" \
    --features.trim_to_first_n_turns 6 \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"


analysis \
    --run_name "analysis/exp4a" \
    --features.min_executed_turns 6 \
    --features.use_embeddings "lexical" \
    --features.use_roles "conversation" "assistant" "user" \
    --features.use_features "l2norm" "catch22" \
    --features.trim_the_last_n_turns 1 \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"



analysis \
    --run_name "analysis/exp4b" \
    --features.min_executed_turns 6 \
    --features.use_embeddings "lexical" \
    --features.use_roles "conversation" "assistant" "user" \
    --features.use_features "l2norm" "catch22" \
    --features.trim_the_last_n_turns 2 \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"


analysis \
    --run_name "analysis/exp4c" \
    --features.min_executed_turns 6 \
    --features.use_embeddings "lexical" \
    --features.use_roles "conversation" "assistant" "user" \
    --features.use_features "l2norm" "catch22" \
    --features.trim_the_last_n_turns 3 \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"

analysis \
    --run_name "analysis/exp4d" \
    --features.min_executed_turns 6 \
    --features.use_embeddings "lexical" \
    --features.use_roles "conversation" "assistant" "user" \
    --features.use_features "l2norm" "catch22" \
    --features.trim_the_last_n_turns 4 \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"

analysis \
    --run_name "analysis/exp4e" \
    --features.min_executed_turns 6 \
    --features.use_embeddings "lexical" \
    --features.use_roles "conversation" "assistant" "user" \
    --features.use_features "l2norm" "catch22" \
    --features.trim_the_last_n_turns 5 \
    --classifiers.logistic_regression.C 1.0 \
    --classifiers.gradient_boosting.learning_rate 0.1 \
    --classifiers.gradient_boosting.max_depth 3 \
    --classifiers.gradient_boosting.l2_regularization 0.01 \
    --dataset_dirs "${dataset_dirs[@]}"