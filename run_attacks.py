import argparse
import asyncio
from datetime import datetime

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from pyrit.setup import initialize_pyrit_async
from pyrit.datasets import SeedDatasetProvider
from src.attacks import *
from src.utils import load_env, setup_experiment, setup_logging


async def initialize_memory(experiment_dir):
    """Initialize PyRIT async with SQLite memory in experiment dir."""
    db_path = experiment_dir / "memory.db"
    await initialize_pyrit_async(memory_db_type="SQLite", db_path=str(db_path))


async def fetch_seeds(seed_cfg):
    """Fetch seed dataset and return (seeds, num_samples)."""
    seed_name = seed_cfg.get("type", "adv_bench")
    num_samples = seed_cfg.get("num_samples", 3)
    seed_dataset = await SeedDatasetProvider.fetch_datasets_async(dataset_names=[seed_name])
    seeds = seed_dataset[0].seeds
    return seeds, num_samples


def build_attack(attack_cfg):
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
    semaphore,
    attack,
    seed,
    idx,
    logger,
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
    max_concurrency,
    seeds,
    num_samples,
    attack,
    logger,
):
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
    cfg,
    experiment_dir,
    logger,
    max_concurrency
):
    """Initialize memory, fetch seeds, build attack, run queue, return results."""

    start_time = datetime.now()

    logger.info("Initializing memory...")
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



def main():

    load_env()

    parser = argparse.ArgumentParser(description="Run red-teaming attacks from a YAML config.")
    parser.add_argument("--root_dir", type=str, default="./experiments", help="Root directory for experiment runs")
    parser.add_argument("--config_path", type=str, default="./configs/datasetV1.yaml", help="Path to YAML config")
    parser.add_argument("--max_concurrency", type=int, default=1, help="Max concurrent attacks")
    
    args = parser.parse_args()
    
    cfg, experiment_dir = setup_experiment(args.root_dir, args.config_path)

    logger = setup_logging(experiment_dir)
    
    asyncio.run(run_attacks(cfg, experiment_dir, logger, args.max_concurrency))

if __name__ == "__main__":
    main()
