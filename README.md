# Psycho-Pass

Red-teaming pipeline: generate multi-turn jailbreak attacks, export conversations to Parquet, compute text embeddings, and evaluate trajectory metrics (velocity, directness, displacement) to compare successful vs failed attacks.

---

## Structure

| Module | Script | Input | Output |
|--------|--------|--------|--------|
| **Attack generation** | `run_attacks.py` | JSON config (e.g. `configs/dataset_base.json`) | Timestamped dir under `--save_to`: `memory.db`, `config.json`, `out.log`, `dataset.parquet` |
| **Dataset processing** | (built into `run_attacks.py`) | `memory.db` (PyRIT SQLite) | `dataset.parquet` (conversations + outcomes) |
| **Embeddings** | `run_encoders.py` | Experiment dir containing `dataset.parquet` + encoder config | `embeddings.parquet` (TF-IDF or other encoder) |
| **Trajectory evaluation** | (built into `run_encoders.py`) | `embeddings.parquet` | `metrics.csv` + comparative report (success vs failure) |

**Helpers**

- `helpers.py`: `describe_db(db_path)`, `view_conversations(db_path, n_conversations)` for inspecting a `memory.db`.
- `tests.py`: PyRIT init + connectivity check against an OpenAI-compatible endpoint (e.g. Ollama).

---

## Setup

### 1. Ollama (optional, for local runs)

```bash
brew install ollama
ollama serve
ollama pull llama3.1
```

Check the OpenAI-compatible API:

```bash
curl -s http://localhost:11434/v1/models | head
```

If you get JSON, the endpoint is up. If you see 404, your Ollama build does not expose the OpenAI-compatible API.

### 2. Python environment (uv)

```bash
uv venv --python 3.12
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uv pip install pyrit==0.10.0
uv pip install tenacity==9.1.2
uv pip install torch==2.10.0
uv pip install scikit-learn==1.8.0
uv pip install pyarrow pandas
```

### 3. Environment variables

For **tests** (OpenAI-compatible endpoint, e.g. Ollama):

```bash
export OPENAI_CHAT_KEY="ollama"
export OPENAI_CHAT_ENDPOINT="http://localhost:11434/v1"
export OPENAI_CHAT_MODEL="llama3.1"
```

For **attack runs**, set the vars that match your config’s `type`:

- **Ollama** (e.g. `configs/dataset_base.json`):
  - `OLLAMA_CHAT_KEY`, `OLLAMA_CHAT_ENDPOINT`
- **Together** (e.g. `configs/dataset_feb3.json`):
  - `TOGETHER_CHAT_KEY`, `TOGETHER_CHAT_ENDPOINT`

Example `.env`:

```bash
OLLAMA_CHAT_KEY="ollama"
OLLAMA_CHAT_ENDPOINT="http://localhost:11434/v1"

TOGETHER_CHAT_KEY=<YOUR_API_KEY>
TOGETHER_CHAT_ENDPOINT="https://api.together.xyz/v1"
```

### 4. Sanity check

```bash
python run_tests.py
```

This creates `./experiments/tests/memory.db` and prints the latest test conversation.

**Common issues**

- “Model 'llama3' not found”: set `OPENAI_CHAT_MODEL` to the exact name from `ollama list` (e.g. `llama3.1:latest`).
- `/v1/*` returns 404: Ollama is not exposing the OpenAI-compatible API.

---

## Run attack generation

Generates multi-turn attacks (RTA), writes PyRIT SQLite + logs, then converts the DB to `dataset.parquet` in the same run.

```bash
python run_attacks.py --save_to ./experiments --config_path ./configs/dataset_base.json --max_concurrency 1
```

- `--save_to`: root directory for results (a timestamped folder is created).
- `--config_path`: dataset/attack JSON config.
- `--max_concurrency`: max concurrent attacks.
- `--debug`: limit to 3 seeds and 1 sample per seed.

**Output** (e.g. `./experiments/dataset_base_<timestamp>/`):

- `memory.db` — PyRIT SQLite
- `config.json` — copied config
- `out.log` — run log
- `dataset.parquet` — standardized conversations + outcomes

**Config (dataset)**  
Example: `configs/dataset_base.json`

- `seed`: e.g. `{"name": "adv_bench", "num_samples": 3}`.
- `attack`: `name` (e.g. `"rta"`), `max_turns`, and for RTA: `adversarial`, `scoring`, `objective` (each with `type`, `model_name`, `temperature`; optional `prompt` for adversarial/scoring).

Ensure the env vars for the `type` you use (ollama/together) are set or in `.env`.

---

## Run embeddings and trajectory evaluation

Reads an experiment folder that already contains `dataset.parquet`, runs the configured encoder (e.g. TF-IDF), writes embeddings, then computes trajectory metrics and a success-vs-failure comparison.

```bash
python run_encoders.py --load_from ./experiments/dataset_base_<timestamp> --config_path ./configs/embeddings_base.json
```

- `--load_from`: path to the experiment directory that contains `dataset.parquet`.
- `--config_path`: encoder config (e.g. `configs/embeddings_base.json`).

**Output** (inside `--load_from`, in a new timestamped subfolder):

- `embeddings.parquet` — one embedding sequence per conversation.
- `metrics.csv` — per-conversation trajectory metrics (velocity, path length, displacement, directness).
- `out.log`, `config.json` — run log and config copy.

**Config (embeddings)**  
Example: `configs/embeddings_base.json`

- `encoder`: e.g. `{"name": "tfidf", "max_features": 1000, "max_df": 0.95, "stop_words": "english"}`.

---

## Inspecting a run

```python
from src.helpers import describe_db, view_conversations

describe_db("./experiments/dataset_base_<timestamp>/memory.db")
view_conversations("./experiments/dataset_base_<timestamp>/memory.db", n_conversations=5)
```

Or from the shell:

```bash
python -c "from helpers import describe_db; describe_db('./experiments/dataset_base_<timestamp>/memory.db')"
```


