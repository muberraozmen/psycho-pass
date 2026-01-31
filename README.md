
## Setup (with Ollama)
1. Install and start Ollama and pull a model. 
```bash
brew install ollama
ollama serve
ollama pull llama3.1
```

Sanity check: the OpenAI-compatible endpoint
```bash
curl -s http://localhost:11434/v1/models | head
```

If this returns JSON, the endpoint is live. If you see 404, your Ollama build/config does not expose the OpenAI-compatible API.

2. Create a Python environment + install packages (uv)
```bash
uv venv --python 3.12.8
source .venv/bin/activate
uv pip install pyrit==0.10.0
```

3. Export the environment variables 
```bash
export OLLAMA_CHAT_KEY="ollama"
export OLLAMA_CHAT_ENDPOINT="http://localhost:11434/v1"
export OLLAMA_CHAT_MODEL="llama3.1"
```

**Tests**
Use `tests.py` to run initialization + connectivity checks.

Run it:
```bash
python tests.py
```

This writes a SQLite DB to `./db/test.db` and prints the latest test conversation.

Common pitfalls
- If you see "model 'llama3' not found", set `OPENAI_CHAT_MODEL` to the exact name from `ollama list` (e.g. `llama3.1:latest`).
- If `/v1/*` returns 404, your Ollama build/config does not expose the OpenAI-compatible API.

## Dataset Generation

Given a config, e.g. `./configs/rta_base.json` use `generate.py` to generate a set of attacks. This script initializes the database, loads config, and generates attacks based on seeds and attack settings.

Example usage:
```bash
python generate.py --db_root ./db --config_path ./configs/rta_base.json
```

- `--db_root` specifies the output directory for results/database (a timestamped folder with `.db` and config snapshot will be created).
- `--config_path` points to your JSON config (see `configs/rta_base.json` for an example).

**Notes**
- Each run creates a new results directory under `db_root` for reproducibility.
- Ensure environment variables are set or provided in `.env` for the corresponding models requested, see `attacks.py/build_chat_model`.


