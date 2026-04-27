# Psycho-Pass

PsychoPass is a research codebase for studying the geometry of multi-turn adversarial LLM conversations.

The pipeline is split into two stages:

1. **Generation**: run PyRIT attacks (`rta` or `crescendo`) and save conversation memory plus lexical/semantic embeddings.
2. **Analysis**: compute trajectory features (L2 + catch22) and train classifiers (logistic regression and gradient boosting).

## Repository layout

| Path | Purpose |
|------|---------|
| `cli.py` | Main entrypoints: `generation()` and `analysis(...)` |
| `src/attacks.py` | Attack wrappers for PyRIT RedTeam and Crescendo |
| `src/encoders.py` | `LexicalEncoder` (TF-IDF) and `SemanticEncoder` (OpenRouter embeddings) |
| `src/features.py` | L2 trajectory features and selected catch22 features |
| `src/classifiers.py` | Model training/evaluation + significant factor extraction |
| `src/factory.py` | Orchestration factories for attack, encoding, feature extraction, and classification |
| `src/utils.py` | Experiment directory creation, config snapshotting, logging |
| `shells/principal.sh` | Reproducible generation sweep across datasets/objective models |
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

## Environment variables

Create `.env` at the project root:

```bash
OPENROUTER_API_KEY=...
OPENROUTER_CHAT_ENDPOINT=https://openrouter.ai/api/v1
```

Both attack generation and semantic embeddings use OpenRouter credentials.

## Stage 1: Generate attack trajectories

After install, the `generation` console script is available from `pyproject.toml`.

Minimal example:

```bash
generation \
  --run_name demo \
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
  --encoders.semantic.model_name qwen/qwen3-embedding-8b
```

For the full principal run used in the project, use:

```bash
bash shells/principal.sh
```

### Generation outputs

Each run creates `experiments/<run_name>/<timestamp>/` with:

- `config.json`: resolved run configuration
- `out.log`: run logs
- `memory.db`: PyRIT SQLite memory
- `embeddings_lexical.parquet`: TF-IDF embeddings per message
- `embeddings_semantic.parquet`: dense embeddings per message

## Stage 2: Feature extraction and classification

Analysis is configured through JSON files in `configs/` (for example `experiment2a.json`, `experiment3b.json`, `tune.json`).

Expected config sections:

- `features`:
  - filters: `min_executed_turns`, `max_executed_turns`
  - trajectory scope: `use_embeddings`, `use_roles`, `use_features`
  - trimming: `trim_to_first_n_turns` or `trim_the_last_n_turns`
- `classifiers`:
  - `logistic_regression` hyperparameter grid (e.g. `C`)
  - `gradient_boosting` hyperparameter grid (`learning_rate`, `max_depth`, `l2_regularization`)

Run analysis from Python:

```python
from cli import analysis

results = analysis(run_name, config_file, dataset_dirs)
```

### Analysis outputs

Analysis writes to a new timestamped experiment directory:

- `features.csv`: conversation-level feature table + `outcome`
- `out.log`: metrics and classifier summaries
- `config.json`: copied analysis config with `run_name`

## Feature set summary

`src/features.py` currently computes:

- **L2 trajectory features**: `distance`, `displacement`, `speed`, `speed_std`, `velocity`, `directness`, `circularity`
- **catch22 subset**: `outlier_timing_pos`, `outlier_timing_neg`, `high_fluctuation`, `stretch_decreasing`, `stretch_high`, `trev`

For catch22, features are computed over top-variance embedding dimensions (`top_k=100`) and aggregated with `mean`, `max`, `min`, `std`.
