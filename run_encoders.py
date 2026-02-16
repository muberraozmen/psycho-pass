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
    def __init__(self, load_from: Path, experiment_dir: Path, cfg: dict):
        self.load_from = load_from
        self.experiment_dir = experiment_dir
        self.cfg = cfg

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

    def _make_encoder(self):
        # Robust config retrieval (handles root vs 'encoder' subkey)
        encoder_cfg = self.cfg.get("encoder", self.cfg)
        encoder_name = encoder_cfg.get("name", "tfidf")
        
        if encoder_name == "tfidf":
            return TFIDFEncoder(encoder_cfg)
        else:
            raise ValueError(f"Unsupported encoder name: {encoder_name}")

    def _calculate_metrics(self):
        # Calculate metrics
        logger.info(f"Evaluating trajectory metrics...")        
        evaluate_trajectory(
            embeddings_path=str(self.experiment_dir) + "/embeddings.parquet", 
            metrics_path=str(self.experiment_dir) + "/metrics.csv"
            )
        logger.info(f"Trajectory metrics saved as {self.experiment_dir}/metrics.csv")

        # Log comparative stats
        logger.info("Summary")
        full_report = log_comparative_stats(metrics_csv_path=str(self.experiment_dir) + "/metrics.csv")
        logger.info(full_report)

    def run(self):
        """Main execution flow."""
        encoder = self._make_encoder()
        logger.info(f"Running encoder...")
        encoder.run(str(self.load_from), str(self.experiment_dir))
        logger.info(f"Embeddings are saved as {self.experiment_dir}/embeddings.parquet")
        self._calculate_metrics()
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from", type=str, default="./experiments/dataset_feb3_1770230537", 
                        help="Path to the experiment folder containing dataset.parquet")
    parser.add_argument("--config_path", type=str, default="./configs/embeddings_base.json")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    # Setup Paths
    timestamp = int(datetime.now().timestamp())
    run_name = f"{config_path.stem}_{timestamp}"
    load_path = Path(args.load_from)
    experiment_dir = load_path / run_name

    # Calculate embeddings
    calculator = EmbeddingsCalculator(
        load_from=load_path,
        experiment_dir=experiment_dir,
        cfg=cfg
    )
    calculator.run()

if __name__ == "__main__":
    main()