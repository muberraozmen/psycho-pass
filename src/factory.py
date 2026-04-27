from __future__ import annotations
from datetime import datetime
from itertools import product
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


MEMORY_FILENAME = "memory.db"
EMBEDDINGS_FILENAMES = {
    "lexical": "embeddings_lexical.parquet",
    "semantic": "embeddings_semantic.parquet",
}
FEATURES_FILENAME = "features.csv"


class AttackFactory:
    def __init__(self, cfg, experiment_dir, logger):
        self.cfg = cfg
        self.experiment_dir = experiment_dir
        self.logger = logger

        self.memory_path = self.experiment_dir / MEMORY_FILENAME
        self.attack_type = cfg.get("attack", {}).get("type")
        self.dataset_names = cfg.get("seeds", {}).get("dataset_names")
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
        datasets = await SeedDatasetProvider.fetch_datasets_async(dataset_names=self.dataset_names)
        seeds = []
        for d in datasets:
            seeds.extend(d.seeds * self.num_samples)
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


class FeaturesFactory:
    def __init__(self, cfg, experiment_dir, logger):
        self.cfg = cfg
        self.experiment_dir = experiment_dir
        self.logger = logger
        
        features_cfg = self.cfg.get("features", {})

        # Filters on attack results
        self.min_executed_turns = features_cfg.get("min_executed_turns", None)
        self.max_executed_turns = features_cfg.get("max_executed_turns", None)

        # Filters on trajectories
        self.use_embeddings = features_cfg.get("use_embeddings", ["semantic", "lexical"])
        self.use_roles = features_cfg.get("use_roles", ["conversation", "assistant", "user"])
        self.use_features = features_cfg.get("use_features", ["l2norm", "executed_turns"])

        # Configuration on trimming (at most one non-None)
        self.trim_to_first_n_turns = features_cfg.get("trim_to_first_n_turns", None)
        self.trim_the_last_n_turns = features_cfg.get("trim_the_last_n_turns", None)
        if (
            self.trim_to_first_n_turns is not None
            and self.trim_the_last_n_turns is not None
        ):
            raise ValueError(
                "Set at most one of trim_to_first_n_turns and trim_the_last_n_turns."
            )

    def load_attack_results(self, dataset_dir):
        with sqlite3.connect(dataset_dir / MEMORY_FILENAME) as conn:
            are = pd.read_sql_query("SELECT * FROM AttackResultEntries;", conn)
        are = are[["conversation_id", "outcome", "executed_turns"]]
        are["outcome"] = are["outcome"].map({"success": 1, "failure": 0})
        are.set_index("conversation_id", inplace=True)
        return are.to_dict(orient="index")

    def filter_by_num_turns(self, trajectories):
        trajectories["sequence_length"] = trajectories.groupby("conversation_id")["conversation_id"].transform("size")
        
        if self.min_executed_turns is not None:
            trajectories = trajectories[trajectories["sequence_length"] >= 2*self.min_executed_turns]
        
        if self.max_executed_turns is not None:
            trajectories = trajectories[trajectories["sequence_length"] <= 2*self.max_executed_turns]

        if self.trim_to_first_n_turns is not None:
            trajectories = trajectories[trajectories["sequence_length"] >= 2*self.trim_to_first_n_turns]
            trajectories = trajectories[trajectories["sequence"] < self.trim_to_first_n_turns * 2]

        if self.trim_the_last_n_turns is not None:
            trajectories = trajectories[trajectories["sequence_length"] > 2*self.trim_the_last_n_turns]
            trajectories = trajectories[trajectories["sequence"] < trajectories["sequence_length"] - self.trim_the_last_n_turns * 2]

        return trajectories

    def load_trajectories(self, dataset_dir, embeddings, role):
        trajectories = pd.read_parquet(dataset_dir / EMBEDDINGS_FILENAMES[embeddings])
        trajectories = trajectories[["conversation_id", "sequence", "role", "embeddings"]]

        trajectories = self.filter_by_num_turns(trajectories)

        if role in ["assistant", "user"]:
            trajectories = trajectories[trajectories["role"] == role]

        trajectories.sort_values(by=["conversation_id", "sequence"], inplace=True)
        trajectories.reset_index(drop=True, inplace=True)

        trajectories = trajectories.groupby("conversation_id")["embeddings"].apply(
            lambda x: np.asarray(x.to_list(), dtype=np.float64))

        return trajectories.to_dict()

    def calculate_features(self, dataset_dir):
        are = self.load_attack_results(dataset_dir)
        data = {}
        for embeddings in self.use_embeddings:
            for role in self.use_roles:
                trajectories = self.load_trajectories(dataset_dir, embeddings, role)
                for conversation_id, traj in trajectories.items():
                    try:
                        result = are[conversation_id]
                        if "executed_turns" not in self.use_features:
                            result.pop("executed_turns")
                        if "l2norm" in self.use_features:
                            feat = l2norm_features(traj)
                            feat = {f"l2norm_{embeddings}_{role}_{k}": v for k, v in feat.items()}
                            result.update(feat)
                        if "catch22" in self.use_features:
                            feat = catch22_features(traj)
                            feat = {f"catch22_{embeddings}_{role}_{k}": v for k, v in feat.items()}
                            result.update(feat)
                        data[conversation_id] = result
                    except KeyError:
                        continue
        data = pd.DataFrame.from_dict(data, orient="index")

        return data

    def summarize_data(self, data):
        self.logger.info(f"Number of samples: {len(data)}")
        self.logger.info(f"Number of features: {len(data.columns)}")
        self.logger.info(f"Success rate: {data['outcome'].mean()}")
        # self.logger.info(f"Mean and standard deviation of features by outcome:")
        # summary = data.groupby("outcome").agg(["mean", "std"]).T.to_string(line_width=10_000)
        # self.logger.info("\n%s", summary)

    def execute(self, dataset_dirs):
        self.logger.info(f"Calculating features for datasets: \n{"\n".join([str(dataset_dir) for dataset_dir in dataset_dirs])}")
        data = []
        for dataset_dir in dataset_dirs:
            data.append(self.calculate_features(dataset_dir))
        data = pd.concat(data)
        self.summarize_data(data)
        data.to_csv(self.experiment_dir / FEATURES_FILENAME)
        self.logger.info(f"Features are saved as: {self.experiment_dir / FEATURES_FILENAME}")


class ClassifierFactory:
    def __init__(self, cfg, experiment_dir, logger):
        self.cfg = cfg
        self.experiment_dir = experiment_dir
        self.logger = logger
        self.classifiers = self.cfg.get("classifiers", {})

    def execute(self):
        results = []

        for classifier_name, hyperparameters in self.classifiers.items():
            data = pd.read_csv(self.experiment_dir / FEATURES_FILENAME, index_col=0)

            X = data.drop(columns=["outcome"])
            y = data["outcome"]

            if classifier_name == "logistic_regression":
                classifier = logistic_regression
            elif classifier_name == "gradient_boosting":
                classifier = gradient_boosting
            else:
                raise ValueError(f"Unsupported classifier name: {classifier_name}")

            performance, factors = classifier(X, y, hyperparameters)
            performance_table = performance.to_string(index=False, line_width=10_000)
            factors_table = factors.to_string(index=False, line_width=10_000)

            self.logger.info(f"Classifier: {classifier.__name__}")
            self.logger.info(f"Hyperparameters: {hyperparameters}")
            self.logger.info("Performance:\n%s", performance_table)
            self.logger.info("Factors:\n%s", factors_table)

            results.append((classifier_name, performance, factors))
        
        return results

