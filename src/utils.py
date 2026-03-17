import yaml
import json
import logging
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import sqlite3
import pandas as pd


def load_env() -> None:
    """Load environment variables from .env file."""
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")


def setup_experiment(root_dir: str | Path, config_path: str | Path) -> (dict, Path):
    """Create timestamped experiment directory under save_root."""
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)

    if isinstance(config_path, str):
        config_path = Path(config_path)

    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Create experiment directory
    experiment_name = config_path.stem
    timestamp = int(datetime.now().timestamp())
    run_name = f"{experiment_name}_{timestamp}"
    experiment_dir = root_dir / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Dump config to experiment directory
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=4)
    
    return cfg, experiment_dir


def setup_logging(experiment_dir: str | Path) -> logging.Logger:
    """Configure root logger with file and console handlers; return script logger."""
    if isinstance(experiment_dir, str):
        experiment_dir = Path(experiment_dir)

    log_file = experiment_dir / "out.log"
    level = logging.INFO

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if root_logger.handlers:
        root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger("pyrit").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    return logging.getLogger(__name__)


def extract_messages(db_path: Path | str) -> pd.DataFrame:

    with sqlite3.connect(db_path) as conn:
        attack_result_entries = pd.read_sql_query("SELECT * FROM AttackResultEntries;", conn)
        prompt_memory_entries = pd.read_sql_query("SELECT * FROM PromptMemoryEntries;", conn)

    df = prompt_memory_entries.merge(
        attack_result_entries,
        on=["conversation_id", "attack_identifier"],
        how="left",
        suffixes=("", "_attack_result"),
    )

    df.rename(columns={
        "id_attack_result": "attack_result_id", 
        "converted_value": "value"}, 
        inplace=True
    )

    df = df[~df["attack_result_id"].isna()]

    columns = [
        "attack_result_id", "conversation_id", "last_score_id",
        "objective", "executed_turns", "outcome", "outcome_reason",
        "id", "role", "sequence", "value"
    ]
    df = df[columns]

    df.sort_values(by=["attack_result_id", "sequence"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.reset_index(inplace=True)

    return df
