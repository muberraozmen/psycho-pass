# Psycho-Pass

End-to-end red-teaming pipeline: run **PyRIT** attacks (RTA or Crescendo) against a chat API, persist results in SQLite, build **lexical** (TF-IDF) and **semantic** (OpenRouter embeddings) trajectories per conversation, then compute **L2 geometry metrics** (path length, displacement, directness, speed statistics, circularity) for each trajectory—split by lexical vs semantic encoders.

A single entrypoint orchestrates all stages: attacks → embeddings → metrics.

---

## Repository layout

| Area | Role |
|------|------|
| `run.py` | CLI: loads YAML config, creates an experiment directory, runs attack → encoder → metrics pipeline |
| `src/attacks.py` | `RTA` and `Crescendo` attacks; OpenAI-compatible chat via PyRIT `OpenAIChatTarget` |
| `src/factory.py` | `AttackFactory` (async attacks, PyRIT memory), `EncoderFactory` (Parquet embeddings), `MetricsFactory` (CSV metrics) |
| `src/encoders.py` | `LexicalEncoder` (scikit-learn TF-IDF), `SemanticEncoder` (OpenRouter embeddings API) |
| `src/metrics.py` | `l2_norm_metrics`: trajectory statistics in embedding space |
| `src/utils.py` | `make_experiment`: merge CLI args into config, create `experiments/…` dir, logging |
| `configs/` | YAML experiment configs (e.g. `base.yaml`, model-specific variants under `model_iterations/`) |

---

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or another PEP 517 installer

---

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
```

This installs dependencies from `pyproject.toml` and locks versions via `uv.lock`.

---

## Environment variables

Create a `.env` file in the project root (`run.py` loads it with `python-dotenv`).

**Chat completions (attacks)** — used by `src/attacks.py` for adversarial, scoring, and objective models:

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | API key for OpenRouter |
| `OPENROUTER_CHAT_ENDPOINT` | OpenAI-compatible base URL (typically `https://openrouter.ai/api/v1` for chat completions) |

**Embeddings** — used by `SemanticEncoder` in `src/encoders.py`:

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | Same key; required for semantic embedding requests |

Ensure your OpenRouter account and models match the `model_name` fields in your YAML (e.g. `meta-llama/llama-3.1-8b-instruct` for chat, `qwen/qwen3-embedding-8b` for embeddings).

---

## Configuration

Configs are YAML files under `configs/`. The CLI flag `--base_config` is resolved relative to `configs/` (default: `base.yaml`).

Typical sections:

- **`seeds`**: `dataset_name` (PyRIT `SeedDatasetProvider`, e.g. `adv_bench`), `num_samples` (replication factor for seeds)
- **`attack`**: `type` (`rta` or `crescendo`), `max_turns`, `scoring_threshold`, optional `max_backtracks` (Crescendo), and nested `adversarial` / `scoring` / `objective` blocks (`model_name`, `temperature`, `top_p`, `max_completion_tokens`, …)
- **`encoders`**: `lexical` (TF-IDF: `max_features`, `stop_words`, `min_df`, `max_df`) and `semantic` (`model_name`, `max_context_tokens` for truncation)
- **`metrics`**: reserved for future toggles; L2 metrics are always computed for both encoder outputs when present

See `configs/base.yaml` and `configs/model_iterations/` for examples.

---

## Run

```bash
python run.py --base_config base.yaml --max_concurrency 3
```

Optional:

- `--run_name my_run` — results go under `experiments/my_run/<timestamp>/`
- Without `--run_name` — `experiments/<timestamp>/`

After `uv sync`, you can also use the installed console script:

```bash
psycho-pass --base_config base.yaml --max_concurrency 3
```

---

## Outputs (per experiment directory)

| File | Contents |
|------|----------|
| `config.json` | Resolved configuration (CLI merged into YAML) |
| `out.log` | Run log |
| `memory.db` | PyRIT SQLite memory (`AttackFactory`) |
| `embeddings.parquet` | Merged prompt/attack rows with `embeddings_lexical` and `embeddings_semantic` columns |
| `metrics.csv` | One row per `conversation_id`: outcomes, roles, sequences, and `lexical_*` / `semantic_*` metric columns from `l2_norm_metrics` |

The `experiments/` directory is ignored by git (see `.gitignore`).

---

## Metrics (embedding trajectories)

For each conversation, message-level embeddings are ordered by `sequence`. `l2_norm_metrics` reports quantities such as total path length, displacement (start→end distance), directness (displacement / path length), per-step speed stats, a velocity term (displacement divided by number of steps), and circularity for trajectories with at least three points. Short conversations (fewer than two points) yield `nan` for most fields.
