from __future__ import annotations
from datetime import datetime
from re import S
import numpy as np
import pandas as pd
import asyncio
import sqlite3
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from pyrit.setup import initialize_pyrit_async
from pyrit.datasets import SeedDatasetProvider
from src.attacks import *
from src.encoders import * 
from src.metrics import l2_norm_metrics


MEMORY_FILENAME = "memory.db"
EMBEDDINGS_FILENAMES = {
    "lexical": "embeddings_lexical.parquet",
    "semantic": "embeddings_semantic.parquet",
}
METRICS_CONFIGS = [
    {"filename": "metrics_lexical.csv", "embeddings": "lexical", "role": None, "trim": None},
    {"filename": "metrics_semantic.csv", "embeddings": "semantic", "role": None, "trim": None},
    {"filename": "metrics_lexical_user.csv", "embeddings": "lexical", "role": "user", "trim": None},
    {"filename": "metrics_semantic_user.csv", "embeddings": "semantic", "role": "user", "trim": None},
    {"filename": "metrics_lexical_assistant.csv", "embeddings": "lexical", "role": "assistant", "trim": None},
    {"filename": "metrics_semantic_assistant.csv", "embeddings": "semantic", "role": "assistant", "trim": None},
    {"filename": "metrics_lexical_trim1.csv", "embeddings": "lexical", "role": None, "trim": 1},
    {"filename": "metrics_semantic_trim1.csv", "embeddings": "semantic", "role": None, "trim": 1},
    {"filename": "metrics_lexical_user_trim1.csv", "embeddings": "lexical", "role": "user", "trim": 1},
    {"filename": "metrics_semantic_user_trim1.csv", "embeddings": "semantic", "role": "user", "trim": 1},
    {"filename": "metrics_lexical_assistant_trim1.csv", "embeddings": "lexical", "role": "assistant", "trim": 1},
    {"filename": "metrics_semantic_assistant_trim1.csv", "embeddings": "semantic", "role": "assistant", "trim": 1},
    {"filename": "metrics_lexical_trim2.csv", "embeddings": "lexical", "role": None, "trim": 2},
    {"filename": "metrics_semantic_trim2.csv", "embeddings": "semantic", "role": None, "trim": 2},
    {"filename": "metrics_lexical_user_trim2.csv", "embeddings": "lexical", "role": "user", "trim": 2},
    {"filename": "metrics_semantic_user_trim2.csv", "embeddings": "semantic", "role": "user", "trim": 2},
    {"filename": "metrics_lexical_assistant_trim2.csv", "embeddings": "lexical", "role": "assistant", "trim": 2},
    {"filename": "metrics_semantic_assistant_trim2.csv", "embeddings": "semantic", "role": "assistant", "trim": 2},
]

class AttackFactory:
    def __init__(self, cfg, experiment_dir, logger):
        self.cfg = cfg
        self.experiment_dir = experiment_dir
        self.logger = logger

        self.memory_path = self.experiment_dir / MEMORY_FILENAME
        self.attack_type = cfg.get("attack", {}).get("type")
        self.dataset_name = cfg.get("seeds", {}).get("dataset_name")
        self.num_samples = cfg.get("seeds", {}).get("num_samples", 1)
        self.max_concurrency = cfg.get("max_concurrency", 3)

    async def initialize_memory(self):
        await initialize_pyrit_async(memory_db_type="SQLite", db_path=str(self.memory_path))

    def build_attack(self):
        attack_cfg = self.cfg.get("attack", {})
        if self.attack_type == "rta":
            return RTA(attack_cfg)
        if self.attack_type == "crescendo":
            return Crescendo(attack_cfg)
        raise ValueError(f"Unsupported attack type: {self.attack_type}")

    async def fetch_seeds(self):
        datasets = await SeedDatasetProvider.fetch_datasets_async(dataset_names=[self.dataset_name])
        seeds = datasets[0].seeds * self.num_samples
        return seeds

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(3),
    )
    async def _attack_task(self, attack, seed, semaphore, idx):
        async with semaphore:
            self.logger.info("Starting attack %s (slot acquired)...", idx)
            start_time = datetime.now()
            result = await attack.execute(seed)
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info("Finished attack %s in %d seconds.", idx, int(duration))
            return result

    def summarize_results(self, results):
        num_attacks = len(results)
        success_count = failure_count = 0
        for result in results:
            try:
                if result.outcome.value == "success":
                    success_count += 1
                elif result.outcome.value == "failure":
                    failure_count += 1
            except Exception as e:
                continue
        self.logger.info(f"Number of attacks: {num_attacks}")
        self.logger.info(f"Success count: {success_count}")
        self.logger.info(f"Failure count: {failure_count}")

    async def execute(self):
        await self.initialize_memory()
        attack = self.build_attack()
        seeds = await self.fetch_seeds()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        tasks = []
        for idx, seed in enumerate(seeds):
            tasks.append(self._attack_task(attack, seed, semaphore, idx))
        self.logger.info("Total attack jobs: %d; max concurrency: %d.",len(tasks), self.max_concurrency)

        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info("Attacks completed in %d seconds.", int(duration))
        self.logger.info("Memory is saved as: %s", self.memory_path)

        self.logger.info("Attack Results Summary:")
        self.summarize_results(results)


class EncoderFactory:
    def __init__(self, cfg, experiment_dir, logger):
        self.cfg = cfg
        self.experiment_dir = experiment_dir
        self.logger = logger
        self.memory_path = self.experiment_dir / MEMORY_FILENAME
        self.embeddings_paths = {
            name: self.experiment_dir / filename 
            for name, filename in EMBEDDINGS_FILENAMES.items()
            }

    def build_encoders(self):
        encoder_cfg = self.cfg.get("encoders", {})
        lexical_encoder = LexicalEncoder(encoder_cfg.get("lexical"))
        semantic_encoder = SemanticEncoder(encoder_cfg.get("semantic"))
        return {"lexical": lexical_encoder, "semantic": semantic_encoder}

    def load_prompt_memory_entries(self):
        with sqlite3.connect(self.memory_path) as conn:
            are = pd.read_sql_query("SELECT * FROM AttackResultEntries;", conn)
            pme = pd.read_sql_query("SELECT * FROM PromptMemoryEntries;", conn)
        return pme[pme["conversation_id"].isin(are["conversation_id"])]

    def execute(self):
        encoders = self.build_encoders()
        for name, encoder in encoders.items():
            try:
                df = self.load_prompt_memory_entries()
                embeddings = encoder.execute(df["converted_value"].to_list())
                df["embeddings"] = embeddings
                df.to_parquet(self.embeddings_paths[name])
                self.logger.info(f"{name} embeddings are saved as: %s", self.embeddings_paths[name])
            except Exception as e:
                self.logger.error(f"Error encoding {name} embeddings: {e}")
                continue


class MetricsFactory:
    def __init__(self, cfg, experiment_dir, logger):
        self.cfg = cfg
        self.experiment_dir = experiment_dir
        self.logger = logger
        self.memory_path = self.experiment_dir / MEMORY_FILENAME
        self.embeddings_paths = {name: self.experiment_dir / filename for name, filename in EMBEDDINGS_FILENAMES.items()}
        self.metrics_configs = [config | {"path": experiment_dir / config["filename"]} for config in METRICS_CONFIGS]
    
    def load_attack_result_entries(self):
        with sqlite3.connect(self.memory_path) as conn:
            are = pd.read_sql_query("SELECT * FROM AttackResultEntries;", conn)
        are.set_index("conversation_id", inplace=True)
        return are.to_dict(orient="index")

    def load_embeddings(self):
        return {name: pd.read_parquet(self.embeddings_paths[name]) for name in EMBEDDINGS_FILENAMES.keys()}

    @staticmethod
    def get_trajectory(df, conversation_id, role=None, trim=None):
        conversation = df[df["conversation_id"] == conversation_id]
        conversation.sort_values(by=["sequence"], inplace=True)
        conversation.reset_index(drop=True, inplace=True)
        if trim is not None:
            if len(conversation) > 2*trim:
                conversation = conversation.iloc[:-2*trim]
            else:
                return None
        if role is not None:
            conversation = conversation[conversation["role"] == role]
        
        if len(conversation) <= 2:
            return None
        
        traj = np.asarray(conversation["embeddings"].to_list(), dtype=np.float64)
        return traj

    def execute(self) -> None:
        embeddings = self.load_embeddings()
        for config in self.metrics_configs:
            try:
                attack_results = self.load_attack_result_entries()
                for conversation_id, result in attack_results.items():
                    traj = self.get_trajectory(
                        df=embeddings[config["embeddings"]], 
                        conversation_id=conversation_id, 
                        role=config["role"], 
                        trim=config["trim"])
                    if traj is not None:
                        metrics = l2_norm_metrics(traj)
                        result.update(metrics)
                attack_results = pd.DataFrame.from_dict(attack_results, orient="index")
                attack_results.reset_index(inplace=True)
                attack_results.rename(columns={"index": "conversation_id"}, inplace=True)
                attack_results.to_csv(config["path"], index=False)
                self.logger.info("Metrics are saved as: %s", config["path"])
            except Exception as e:
                self.logger.error(f"Error calculating metrics for {config['filename']}: {e}")
                continue