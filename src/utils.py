from __future__ import annotations

import argparse
from pathlib import Path
import json
from datetime import datetime
import logging
import warnings


def _set_config_value(cfg: dict, key: str, value) -> None:
    """Set the value at dotted ``key`` in ``cfg``."""
    if "." not in key:
        cfg[key] = value
        return
    head, rest = key.split(".", 1)
    if head not in cfg or not isinstance(cfg[head], dict):
        cfg[head] = {}
    _set_config_value(cfg[head], rest, value)


def _get_config_value(cfg, key: str):
    """Return the value at dotted ``key``, or ``None`` if ``cfg`` is not a mapping or any segment is missing."""
    if "." not in key:
        if isinstance(cfg, dict):
            return cfg.get(key)
        return None
    head, rest = key.split(".", 1)
    if not isinstance(cfg, dict):
        return None
    return _get_config_value(cfg.get(head), rest)


def _get_logger(experiment_dir: str | Path):
    """Get a logger for the experiment directory."""
    if isinstance(experiment_dir, str):
        experiment_dir = Path(experiment_dir)

    level = logging.INFO
    log_file = experiment_dir / "out.log"
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

    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("pyrit").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    return logging.getLogger(__name__)


def make_experiment(args: argparse.Namespace):
    """Make an experiment directory and return the configuration, experiment directory, and logger."""
    project_root = Path(__file__).resolve().parent.parent

    cfg = {}
    for k, v in vars(args).items():
        if v is None:
            continue
        _set_config_value(cfg, k, v)

    timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
    if cfg.get("run_name") is not None:
        experiment_dir = project_root / "experiments" / cfg.get("run_name")
    else:
        experiment_dir = project_root / "experiments" / timestamp
    
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(experiment_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    logger = _get_logger(experiment_dir)

    return cfg, experiment_dir, logger
