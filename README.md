# Psycho-Pass

Psycho-Pass is a research codebase for studying the geometry of multi-turn adversarial LLM conversations.

The workflow has two stages:

1. **Generation**: run PyRIT attacks (`rta` or `crescendo`) and save conversation memory plus lexical/semantic embeddings.
2. **Analysis**: build trajectory features (L2 norm + catch22 + optional executed turns) and train classifiers (logistic regression and gradient boosting).

## Repository Layout

| Path | Purpose |
|------|---------|
| `cli.py` | CLI entrypoints: `generation` and `analysis` |
| `src/attacks.py` | Attack wrappers for PyRIT RedTeam and Crescendo |
| `src/encoders.py` | `LexicalEncoder` (TF-IDF) and `SemanticEncoder` (OpenRouter embeddings) |
| `src/features.py` | L2 trajectory features and selected catch22 features |
| `src/classifiers.py` | Classifier training/evaluation + factor extraction |
| `src/factory.py` | Orchestration for attacks, encoders, features, and classifiers |
| `src/utils.py` | Experiment directory creation, config snapshotting, logging |
| `shells/generation.sh` | Reproducible generation sweep across datasets/objective models |
| `shells/analysis.sh` | Reproducible analysis experiment suite |
| `notebooks/` | Exploration and analysis notebooks |

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended)

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

## Environment Variables

Create `.env` at the project root:

```bash
OPENROUTER_API_KEY=...
OPENROUTER_CHAT_ENDPOINT=https://openrouter.ai/api/v1
```

OpenRouter credentials are used by both attack generation and semantic embeddings.

## Run Generation

After install, the `generation` console script is available via `pyproject.toml`.

Minimal example:

```bash
generation \
  --run_name generation/demo \
  --max_concurrency 16 \
  --seeds.dataset_names adv_bench harmbench \
  --seeds.num_samples 1 \
  --attack.type crescendo \
  --attack.max_turns 8 \
  --attack.max_backtracks 2 \
  --attack.scoring_threshold 0.8 \
  --attack.adversarial.model_name meta-llama/llama-3.1-8b-instruct \
  --attack.scoring.model_name openai/gpt-oss-120b \
  --attack.objective.model_name meta-llama/llama-3.1-8b-instruct \
  --encoders.lexical.max_features 4096 \
  --encoders.lexical.stop_words english \
  --encoders.lexical.max_df 0.95 \
  --encoders.semantic.model_name qwen/qwen3-embedding-8b \
  --encoders.semantic.max_context_tokens 32000 \
  --encoders.semantic.batch_size 128
```

Run the full generation sweep used in this repo:

```bash
bash shells/generation.sh
```

### Generation Outputs

Each run writes to `experiments/<run_name>/`:

- `config.json`: resolved run configuration
- `out.log`: run logs
- `memory.db`: PyRIT SQLite memory
- `embeddings_lexical.parquet`: TF-IDF embeddings per message
- `embeddings_semantic.parquet`: dense embeddings per message

## Run Analysis

The `analysis` CLI takes feature/classifier options directly as flags plus one or more `--dataset_dirs`.

Example:

```bash
analysis \
  --run_name analysis/exp1a \
  --features.use_embeddings semantic lexical \
  --features.use_roles conversation assistant user \
  --features.use_features l2norm catch22 executed_turns \
  --classifiers.logistic_regression.C 1.0 \
  --classifiers.gradient_boosting.learning_rate 0.1 \
  --classifiers.gradient_boosting.max_depth 3 \
  --classifiers.gradient_boosting.l2_regularization 0.01 \
  --dataset_dirs \
    experiments/generation/gpt-oss-120b \
    experiments/generation/llama-3.1-8b-instruct
```

Common feature options:

- Filtering: `--features.min_executed_turns`, `--features.max_executed_turns`
- Scope: `--features.use_embeddings`, `--features.use_roles`, `--features.use_features`
- Trimming (mutually exclusive): `--features.trim_to_first_n_turns` or `--features.trim_the_last_n_turns`

Run the full analysis suite:

```bash
bash shells/analysis.sh
```

### Analysis Outputs

Each analysis run writes to `experiments/<run_name>/`:

- `features.csv`: conversation-level feature table with `outcome`
- `out.log`: metrics and classifier summaries
- `config.json`: resolved analysis configuration

## Feature Set Summary

`src/features.py` currently computes:

- **L2 trajectory features**: `distance`, `displacement`, `speed`, `speed_std`, `velocity`, `directness`, `circularity`
- **catch22 subset**: `outlier_timing_pos`, `outlier_timing_neg`, `high_fluctuation`, `stretch_decreasing`, `stretch_high`, `trev`

For catch22, features are computed over top-variance embedding dimensions (`top_k=100`) and aggregated with `mean`, `max`, `min`, and `std`.
