import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from pyrit.setup import initialize_pyrit_async
from pyrit.datasets import SeedDatasetProvider
from src.attacks import *
from src.processors import memory2parquet


import logging
logger = logging.getLogger(__name__)
logging.getLogger("pyrit").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR) 


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(3)
)
async def _execute_single_attack(semaphore, attack, seed, idx):
    """
    Executes a single attack iteration with retry logic.
    Standalone function to avoid 'self' binding issues with @retry.
    """
    async with semaphore:
        logger.info(f"Starting attack {idx}...")
        start_time = datetime.now()
        
        # Execute the attack
        result = await attack.run(seed)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Finished attack {idx} in {int(duration)} seconds.")
        return result


class DatasetGenerator:
    def __init__(self, experiment_dir: Path, cfg: dict, max_concurrency: int = 1, debug: bool = False):
        self.experiment_dir = experiment_dir
        self.cfg = cfg
        self.max_concurrency = max_concurrency
        self.debug = debug
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)        
        self._setup_logging()
        self._dump_config()
    
    def _setup_logging(self):
        """Configures logging to file and console."""
        log_file = self.experiment_dir / "out.log"
        
        # 1. Create your Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 2. Get the Root Logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # 3. Clear existing handlers 
        if root_logger.handlers:
            root_logger.handlers = []
            
        # 4. Add File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # 5. Add Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    def _dump_config(self):
        """Saves the configuration to the experiment directory."""
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(self.cfg, f, indent=4)

    async def _initialize_memory(self):
        """Initializes the PyRIT SQLite memory."""
        db_path = str(self.experiment_dir / "memory.db")
        await initialize_pyrit_async(memory_db_type="SQLite", db_path=db_path)

    async def _fetch_seeds(self):
        """Fetches the attack seeds."""
        seed_cfg = self.cfg.get("seed", {})
        seed_name = seed_cfg.get("name", "adv_bench")
        num_samples = seed_cfg.get("num_samples", 3)

        seed_dataset = await SeedDatasetProvider.fetch_datasets_async(dataset_names=[seed_name])
        seeds = seed_dataset[0].seeds

        if self.debug:
            logger.info("-" * 40)
            logger.info("DEBUG MODE: Limiting to 3 seeds and 1 sample/seed")
            logger.info("-" * 40)
            seeds = seeds[:3]
            num_samples = 1

        return seeds, num_samples

    def _build_attack(self):
        """Instantiates the attack class based on config."""
        attack_cfg = self.cfg.get("attack", {})
        attack_name = attack_cfg.get("name")
        
        if attack_name == "rta":
            return RTA(attack_cfg) 
        else:
            raise ValueError(f"Unsupported attack name: {attack_name}")

    def _create_tasks(self, seeds, num_samples, attack):
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = []
        counter = 0

        logger.info("-" * 40)
        logger.info(f"BATCH STARTING")
        logger.info(f"Seed Count: {len(seeds)} | Samples per Seed: {num_samples}")
        logger.info(f"Max Concurrency: {self.max_concurrency}")
        logger.info(f"Output: {self.experiment_dir}")
        logger.info("-" * 40)

        for seed in seeds:
            for _ in range(num_samples):
                counter += 1
                tasks.append(
                    _execute_single_attack(semaphore, attack, seed, counter)
                )

        return tasks

    def _summarize_results(self, results):
        """Logs the final statistics of the batch."""
        stats = {
            "success": 0,
            "failure": 0,
            "unknown": 0,
            "errors": 0
        }

        for res in results:
            if isinstance(res, Exception):
                stats["errors"] += 1
                logger.info(f"Attack failed with error: {res}")
                continue
            
            # Robust outcome checking (Handles String vs Enum)
            try:
                outcome = str(res.outcome.value).lower() if hasattr(res.outcome, "value") else str(res.outcome).lower()
                
                if "success" in outcome or "true" in outcome:
                    stats["success"] += 1
                elif "fail" in outcome or "false" in outcome:
                    stats["failure"] += 1
                else:
                    stats["unknown"] += 1
            except AttributeError:
                stats["unknown"] += 1

        logger.info("-" * 40)
        logger.info(f"BATCH COMPLETE")
        logger.info(f"Total Attempts: {len(results)}")
        logger.info(f"  [+] Success (Jailbreaks): {stats['success']}")
        logger.info(f"  [-] Failed (Refusals):    {stats['failure']}")
        logger.info(f"  [?] Unknown outcomes:     {stats['unknown']}")
        logger.info(f"  [!] System Errors:        {stats['errors']}")
        logger.info("-" * 40)

    def _process_output(self):
        """Convert memory.db to parquet dataset"""
        
        memory2parquet(
            memory_db_path=self.experiment_dir / "memory.db", 
            parquet_path=self.experiment_dir / "dataset.parquet"
            )
        logger.info(f"The dataset is dumped as {self.experiment_dir}/dataset.parquet.")
    
    async def run(self):
        """Main execution flow."""
        await self._initialize_memory()
        seeds, num_samples = await self._fetch_seeds()
        attack = self._build_attack()        
        tasks = self._create_tasks(seeds, num_samples, attack)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self._summarize_results(results)
        self._process_output()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", type=str, default="./experiments")
    parser.add_argument("--config_path", type=str, default="./configs/dataset_base.json")
    parser.add_argument("--max_concurrency", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    timestamp = int(datetime.now().timestamp())
    run_name = f"{config_path.stem}_{timestamp}"
    experiment_dir = Path(args.save_to) / run_name

    generator = DatasetGenerator(
        experiment_dir=experiment_dir,
        cfg=cfg,
        max_concurrency=args.max_concurrency,
        debug=args.debug
    )

    asyncio.run(generator.run())

if __name__ == "__main__":
    main()