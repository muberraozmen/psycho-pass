import argparse
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from pyrit.setup import initialize_pyrit_async
from pyrit.datasets import SeedDatasetProvider
from src.attacks import *

logger = logging.getLogger(__name__)

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING) 
)
async def run_managed_attack(semaphore, attack, seed, idx):
    async with semaphore:
        logger.warning(f"Starting attack {idx}...")
        start_time = datetime.now()
        result = await attack.run(seed)
        end_time = datetime.now()
        duration = end_time - start_time
        logger.warning(f"Finished attack {idx} in {int(duration.total_seconds())} seconds.")
        return result


async def generate_attacks(
    db_path: str,
    cfg: dict,
    max_concurrency: int = 1, 
    ) -> None:

    # Step 1: Initialize DB
    await initialize_pyrit_async(memory_db_type="SQLite", db_path=db_path)

    # Step 2: Fetch Dataset
    seed_name = cfg.get("seed", {}).get("name", "adv_bench")
    seed_dataset = await SeedDatasetProvider.fetch_datasets_async(dataset_names=[seed_name])
    seeds = seed_dataset[0].seeds[:15]  ## TODO: remove debugger filter
    num_samples = cfg.get("seed", {}).get("num_samples", 3)

    # Step 3: Build Attack
    if cfg["attack"].get("name") == "rta":
        attack = RTA(cfg["attack"])
    else:
        raise ValueError(f"Unsupported attack name: {cfg['attack']['name']}")

    # Step 4: Build Queue with Semaphore
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = []
    counter = 0
    
    for seed in seeds: 
        for _ in range(num_samples):
            counter += 1
            tasks.append(run_managed_attack(semaphore, attack, seed, counter))
    
    # Step 5: Run
    logger.warning(f"Total attacks to generate: {len(tasks)} with max concurrency: {max_concurrency}.")
    logger.warning(f"Destination: {db_path}")
    
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Step 6: Summary
    system_errors = 0
    successful_attacks = 0
    failed_attacks = 0
    unknown_attacks = 0

    for result in results:
        if isinstance(result, Exception):
            system_errors += 1
        elif str(result.score.score_value).lower() == "true":
            successful_attacks += 1
        elif str(result.score.score_value).lower() == "false":
            failed_attacks += 1
        else:
            unknown_attacks += 1

    logger.warning("-" * 40)
    logger.warning(f"BATCH COMPLETE")
    logger.warning(f"Total Attempts: {len(results)}")
    logger.warning(f"  [+] # of Successful Attacks: {successful_attacks}")
    logger.warning(f"  [-] # of Failed Attacks:     {failed_attacks}")
    logger.warning(f"  [?] # of Unknown Attacks:    {unknown_attacks}")
    logger.warning(f"  [!] # of System Errors:      {system_errors}")
    logger.warning("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_root", type=str, default="./db")
    parser.add_argument("--config_path", type=str, default="./configs/rta_base.json")
    parser.add_argument("--max_concurrency", type=int, default=1)

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = json.load(f)

    timestamp = int(datetime.now().timestamp())
    run_name = f"{cfg['attack']['name']}_{cfg['seed']['name']}_{timestamp}"
    db_dir = Path(args.db_root) / run_name
    db_dir.mkdir(parents=True, exist_ok=True)
    
    with open(db_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(db_dir / "generate_attacks.log"),
            logging.StreamHandler()
        ]
    )

    asyncio.run(generate_attacks(
        cfg=cfg,
        db_path=str(db_dir / "attacks.db"),
        max_concurrency=args.max_concurrency,
    ))