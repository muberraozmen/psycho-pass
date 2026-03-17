import argparse
from datetime import datetime

from src.encoders import * 
from src.utils import *


def build_encoder(cfg: dict):
    """Builds encoder based on config."""
    encoder_type = cfg.get("type", "tfidf")
    if encoder_type == "tfidf":
        return TFIDFEncoder(cfg)
    elif encoder_type == "transformers":
        return TransformersEncoder(cfg)
    elif encoder_type == "together":
        return TogetherEncoder(cfg)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")


def run_encoder(
    cfg,
    experiment_dir,
    logger, 
    data
):
    """Runs the encoder on the data and saves the embeddings."""
    start_time = datetime.now()

    logger.info("Building encoder...")
    encoder = build_encoder(cfg)

    logger.info("Running encoder...")
    data["embeddings"] = encoder.run(data["value"].to_list())

    logger.info("Saving embeddings...")
    data.to_parquet(experiment_dir / "embeddings.parquet")

    duration = (datetime.now() - start_time).total_seconds()
    logger.info("Encoding completed in %d seconds.", int(duration))
    logger.info("Embeddings saved as %s/embeddings.parquet", experiment_dir)


def main():
    load_env()
    parser = argparse.ArgumentParser(description="Calculate embeddings from a YAML config.")
    parser.add_argument("--root_dir", type=str, default="./experiments/datasetV1_1772362234", help="Root directory for dataset to calculate embeddings for")
    parser.add_argument("--config_path", type=str, default="./configs/embeddingsV3.yaml", help="Path to the YAML config")

    args = parser.parse_args()
    
    cfg, experiment_dir = setup_experiment(args.root_dir, args.config_path)
    
    logger = setup_logging(experiment_dir)
    
    data = extract_messages(args.root_dir + "/memory.db")
    
    run_encoder(cfg, experiment_dir, logger, data)


if __name__ == "__main__":
    main()