import argparse
import json 
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from src.encoders import * 
from src.metrics import evaluate_trajectory, log_comparative_stats

import pdb

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class EmbeddingsCalculator:
    def __init__(self, load_from: Path, save_to: Path, cfg: dict):
        self.load_from = load_from
        self.save_to = save_to
        self.cfg = cfg

    def _dump_config(self):
        with open(self.save_to / "config.json", "w") as f:
            json.dump(self.cfg, f, indent=4)

    def _make_encoder(self):
        # Robust config retrieval (handles root vs 'encoder' subkey)
        encoder_cfg = self.cfg.get("encoder", self.cfg)
        encoder_name = encoder_cfg.get("name", "tfidf")
        
        if encoder_name == "tfidf":
            return TFIDFEncoder(encoder_cfg)
        else:
            raise ValueError(f"Unsupported encoder name: {encoder_name}")

    def _summarize_results(self):

        pass

    def run(self):
        encoder = self._make_encoder()
        
        logger.info(f"Running encoder...")
        encoder.run(str(self.load_from), str(self.save_to))
        logger.info(f"Embeddings are saved as {self.save_to}/embeddings.parquet")
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from", type=str, default="./experiments/dataset_feb3_1770230537", 
                        help="Path to the experiment folder containing dataset.parquet")
    parser.add_argument("--config_path", type=str, default="./configs/embeddings_base.json")
    args = parser.parse_args()

    # Load Config
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    # Setup Paths
    timestamp = int(datetime.now().timestamp())
    run_name = f"{config_path.stem}_{timestamp}"
    
    # The output folder is created INSIDE the experiment folder
    load_path = Path(args.load_from)
    save_to = load_path / run_name

    # Setup Logging
    save_to.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(save_to / "out.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S'))
    logger.addHandler(fh)

    # Calculate embeddings
    calculator = EmbeddingsCalculator(
        load_from=load_path,
        save_to=save_to,
        cfg=cfg
    )
    calculator.run()

    # Calculate metrics
    logger.info(f"Evaluating trajectory metrics...")        
    evaluate_trajectory(
        embeddings_path=str(save_to) + "/embeddings.parquet", 
        metrics_path=str(save_to) + "/metrics.csv"
        )
    logger.info(f"Trajectory metrics saved as {save_to}/metrics.csv")

    # Log comparative stats
    logger.info("Summary")
    full_report = log_comparative_stats(metrics_csv_path=str(save_to) + "/metrics.csv")
    logger.info(full_report)


if __name__ == "__main__":
    main()