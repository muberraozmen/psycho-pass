from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from datetime import datetime
import logging
import json


def make_experiment(args: argparse.Namespace):

    project_root = Path(__file__).resolve().parent.parent

    # Parse args into config
    with open(project_root / "configs" / args.config_file, "r") as f:
            cfg = yaml.safe_load(f)

    for k, v in args.__dict__.items():
        if v is not None:
            cfg[k] = v

    timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
    if args.run_name is not None:
        experiment_dir = project_root / "experiments" / args.run_name / timestamp
    else:
        experiment_dir = project_root / "experiments" / timestamp

    cfg["experiment_dir"] = str(experiment_dir)
    
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(experiment_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    logger = _get_logger(experiment_dir)

    return cfg, experiment_dir, logger


def _get_logger(experiment_dir: str | Path):
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

    return logging.getLogger(__name__)

