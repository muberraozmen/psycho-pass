import argparse
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime

from typing import Any

from dotenv import load_dotenv

import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from pyrit.setup import initialize_pyrit_async
from pyrit.datasets import SeedDatasetProvider
from src.attacks import *


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config from path."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_experiment_dir(save_root: Path, config_stem: str) -> Path:
    """Create timestamped experiment directory under save_root."""
    timestamp = int(datetime.now().timestamp())
    run_name = f"{config_stem}_{timestamp}"
    experiment_dir = save_root / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def setup_logging(experiment_dir: Path) -> logging.Logger:
    """Configure root logger with file and console handlers; return script logger."""
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


async def initialize_memory(experiment_dir: Path) -> None:
    """Initialize PyRIT async with SQLite memory in experiment dir."""
    db_path = experiment_dir / "memory.db"
    await initialize_pyrit_async(memory_db_type="SQLite", db_path=str(db_path))


async def fetch_seeds(seed_cfg: dict[str, Any]) -> tuple[list, int]:
    """Fetch seed dataset and return (seeds, num_samples)."""
    seed_name = seed_cfg.get("name", "adv_bench")
    num_samples = seed_cfg.get("num_samples", 3)
    seed_dataset = await SeedDatasetProvider.fetch_datasets_async(dataset_names=[seed_name])
    seeds = seed_dataset[0].seeds
    return seeds, num_samples


def build_attack(attack_cfg: dict[str, Any]):
    """Build attack instance from config."""
    attack_type = attack_cfg.get("type")
    if attack_type == "rta":
        return RTA(attack_cfg)
    if attack_type == "crescendo":
        return Crescendo(attack_cfg)
    raise ValueError(f"Unsupported attack type: {attack_type}")


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(3),
)
async def execute_single_attack(
    semaphore: asyncio.Semaphore,
    attack,
    seed,
    idx: int,
    logger: logging.Logger,
):
    """Run one attack with semaphore; retry on exception."""
    async with semaphore:
        logger.info("Starting attack %s...", idx)
        start_time = datetime.now()
        result = await attack.execute(seed)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("Finished attack %s in %d seconds.", idx, int(duration))
        return result


def build_queue(
    max_concurrency: int,
    seeds: list,
    num_samples: int,
    attack,
    logger: logging.Logger,
) -> list:
    """Build list of coroutines for all (seed, sample) combinations."""
    semaphore = asyncio.Semaphore(max_concurrency)
    queue = []
    counter = 0
    for seed in seeds:
        for _ in range(num_samples):
            counter += 1
            queue.append(
                execute_single_attack(semaphore, attack, seed, counter, logger)
            )
    return queue


async def run_attacks(
    experiment_dir: Path,
    cfg: dict[str, Any],
    max_concurrency: int,
    logger: logging.Logger,
):
    """Initialize memory, fetch seeds, build attack, run queue, return results."""

    logger.info("Initializing memory...")
    start_time = datetime.now()
    await initialize_memory(experiment_dir)

    logger.info("Fetching seeds...")
    seeds, num_samples = await fetch_seeds(cfg.get("seed", {}))

    attack = build_attack(cfg.get("attack", {}))
    
    queue = build_queue(max_concurrency, seeds, num_samples, attack, logger)
    
    logger.info("Total number of attacks: %d, starting...", len(queue))
    results = await asyncio.gather(*queue, return_exceptions=True)

    duration = (datetime.now() - start_time).total_seconds()
    logger.info("Attacks completed in %d seconds.", int(duration))
    logger.info("Experiment directory: %s", experiment_dir)



def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    parser = argparse.ArgumentParser(description="Run red-teaming attacks from a YAML config.")
    parser.add_argument("--experiment_dir", type=str, default="./experiments", help="Root directory for experiment runs")
    parser.add_argument("--config_path", type=str, default="./configs/dataset_base.yaml", help="Path to YAML config")
    parser.add_argument("--max_concurrency", type=int, default=1, help="Max concurrent attacks")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    cfg = load_config(config_path)
    
    experiment_dir = setup_experiment_dir(Path(args.experiment_dir), config_path.stem)

    with open(experiment_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    logger = setup_logging(experiment_dir)
    
    asyncio.run(run_attacks(experiment_dir, cfg, args.max_concurrency, logger))


if __name__ == "__main__":
    main()
