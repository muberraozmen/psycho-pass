from __future__ import annotations
import json
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
EMBEDDINGS_FILENAME = "embeddings.parquet"
METRICS_FILENAME = "metrics.csv"


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
        return seeds[:10]

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
        self.embeddings_path = self.experiment_dir / EMBEDDINGS_FILENAME


    def build_encoders(self):
        encoder_cfg = self.cfg.get("encoders", {})
        lexical_encoder = LexicalEncoder(encoder_cfg.get("lexical"))
        semantic_encoder = SemanticEncoder(encoder_cfg.get("semantic"))
        return {"lexical": lexical_encoder, "semantic": semantic_encoder}

    def load_memory(self):
        with sqlite3.connect(self.memory_path) as conn:
            attack_result_entries = pd.read_sql_query("SELECT * FROM AttackResultEntries;", conn)
            prompt_memory_entries = pd.read_sql_query("SELECT * FROM PromptMemoryEntries;", conn)

        df = prompt_memory_entries.merge(
            attack_result_entries,
            on=["conversation_id"],
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

    def execute(self):
        encoders = self.build_encoders()
        dataset = self.load_memory()

        for name, encoder in encoders.items():
            try:
                embeddings = encoder.execute(dataset["value"].to_list())
                dataset[f"embeddings_{name}"] = embeddings
            except Exception as e:
                self.logger.error(f"Error encoding {name} embeddings: {e}")
                continue
        dataset.to_parquet(self.embeddings_path)

        self.logger.info("Embeddings are saved as: %s", self.embeddings_path)


class MetricsFactory:
    def __init__(self, cfg, experiment_dir, logger):
        self.cfg = cfg
        self.experiment_dir = experiment_dir
        self.logger = logger
        self.embeddings_path = self.experiment_dir / EMBEDDINGS_FILENAME
        self.metrics_path = self.experiment_dir / METRICS_FILENAME

    def execute(self) -> None:
        data = pd.read_parquet(self.embeddings_path)

        rows: list[dict] = []

        for conversation_id in data["conversation_id"].unique():
            conversation = data[data["conversation_id"] == conversation_id]
            conversation = conversation.sort_values("sequence")

            metrics = {
                "conversation_id": conversation_id,
                "outcome": conversation["outcome"].iloc[-1]
            }

            executed_turns = conversation["executed_turns"].iloc[-1]
            for name, embeddings in [("lexical", "embeddings_lexical"), ("semantic", "embeddings_semantic")]:
                embeddings = np.asarray(conversation[embeddings].tolist(), dtype=np.float64)
                metrics.update(l2_norm_metrics(embeddings, prefix=name))

                user_embeddings = embeddings[0::2, :]
                metrics.update(l2_norm_metrics(user_embeddings, prefix=f"{name}_user"))

                assistant_embeddings = embeddings[1::2, :]
                metrics.update(l2_norm_metrics(assistant_embeddings, prefix=f"{name}_assistant"))

                if executed_turns >= 2:
                    early_embeddings = embeddings[:-2]
                    metrics.update(l2_norm_metrics(early_embeddings, prefix=f"{name}_early"))

                    user_early_embeddings = early_embeddings[0::2, :]
                    metrics.update(l2_norm_metrics(user_early_embeddings, prefix=f"{name}_user_early"))

                    assistant_early_embeddings = early_embeddings[1::2, :]
                    metrics.update(l2_norm_metrics(assistant_early_embeddings, prefix=f"{name}_assistant_early"))

                if executed_turns >= 3:
                    very_early_embeddings = embeddings[:-4]
                    metrics.update(l2_norm_metrics(very_early_embeddings, prefix=f"{name}_very_early"))

                    user_very_early_embeddings = very_early_embeddings[0::2, :]
                    metrics.update(l2_norm_metrics(user_very_early_embeddings, prefix=f"{name}_user_very_early"))

                    assistant_very_early_embeddings = very_early_embeddings[1::2, :]
                    metrics.update(l2_norm_metrics(assistant_very_early_embeddings, prefix=f"{name}_assistant_very_early"))


            rows.append(metrics)

        pd.DataFrame(rows).to_csv(self.metrics_path, index=False)
        self.logger.info("Metrics are saved as: %s", self.metrics_path)


# TODO: make dataclass for saving embeddings and calculating metrics
# TODO: make early metrics calculation iterative 