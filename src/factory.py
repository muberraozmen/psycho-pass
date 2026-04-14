from __future__ import annotations
from datetime import datetime
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
from src.features import *
from src.classifiers import *

import pdb

MEMORY_FILENAME = "memory.db"
EMBEDDINGS_FILENAMES = {
    "lexical": "embeddings_lexical.parquet",
    "semantic": "embeddings_semantic.parquet",
}


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
        # TODO: Add support for multiple datasets
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


class ClassifierFactory:
    def __init__(self, args, run_dir):
        self.run_dir = run_dir
        self.min_turns = args.get("min_turns", None)
        self.max_turns = args.get("max_turns", None)
        self.embeddings = args.get("embeddings", ["semantic", "lexical"])
        self.roles = args.get("roles", ["conversation", "assistant", "user"])
        self.trim = args.get("trim", None)
        self.features = args.get("features", ["l2norm", "catch22", "executed_turns"])

    def load_attack_result_entries(self, experiment_dir):
        with sqlite3.connect(experiment_dir / MEMORY_FILENAME) as conn:
            are = pd.read_sql_query("SELECT * FROM AttackResultEntries;", conn)
        are = are[["conversation_id", "executed_turns", "outcome"]]
        
        are["outcome"] = are["outcome"].map({"success": 1, "failure": 0})
        
        if self.min_turns is not None:
            are = are[are["executed_turns"] >= self.min_turns]
        
        if self.max_turns is not None:
            are = are[are["executed_turns"] <= self.max_turns]

        are.set_index("conversation_id", inplace=True)
    
        return are.to_dict(orient="index")
    
    def load_trajectories(self, experiment_dir, embeddings="semantic", role="conversation", trim=None):
        trajectories = pd.read_parquet(experiment_dir / EMBEDDINGS_FILENAMES[embeddings])
        trajectories = trajectories[["conversation_id", "sequence", "role", "embeddings"]]
        
        if role in ["assistant", "user"]:
            trajectories = trajectories[trajectories["role"] == role]
        
        if trim is not None:
            trajectories = trajectories[trajectories["sequence"] < int(trim * 2)]
        
        trajectories.sort_values(by=["conversation_id", "sequence"], inplace=True)
        trajectories.reset_index(drop=True, inplace=True)
        
        trajectories = trajectories.groupby("conversation_id")["embeddings"].apply(
            lambda x: np.asarray(x.to_list(), dtype=np.float64))

        return trajectories.to_dict()

    def compute_features(self, experiment_dir):
        are = self.load_attack_result_entries(experiment_dir)
        data = {}
        for embeddings in self.embeddings:
            for role in self.roles:
                trajectories = self.load_trajectories(experiment_dir, embeddings, role, self.trim)
                for conversation_id, result in are.items():
                    try:
                        traj = trajectories[conversation_id]
                        if "l2norm" in self.features:
                            feat = l2norm_features(traj)
                            feat = {f"l2norm_{embeddings}_{role}_{k}": v for k, v in feat.items()}
                            result.update(feat)
                        if "catch22" in self.features:
                            feat = catch22_features(traj)
                            feat = {f"catch22_{embeddings}_{role}_{k}": v for k, v in feat.items()}
                            result.update(feat)
                        data[conversation_id] = result
                    except KeyError:
                        continue
        data = pd.DataFrame.from_dict(data, orient="index")
        if "executed_turns" not in self.features:
            data = data.drop(columns=["executed_turns"])
        return data

    def execute(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = []
        for experiment_dir in [x for x in self.run_dir.glob('*') if x.is_dir()]:
            try:
                d = self.compute_features(experiment_dir)
                data.append(d)
            except Exception:
                continue
        data = pd.concat(data)
        performance, factors = logistic_regression(data.drop(columns=['outcome']), data['outcome'])
        return data,performance, factors
