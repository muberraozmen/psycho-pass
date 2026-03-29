from dotenv import load_dotenv
load_dotenv(".env")

from src.factory import EncoderFactory, MetricsFactory

from pathlib import Path
import json
import logging 

experiments = [
    "/Users/muberraozmen/Development/psycho-pass/experiments/model_iterations/gpt-5.4/17741314440",
    "/Users/muberraozmen/Development/psycho-pass/experiments/model_iterations/nemotron-3-super-120b-a12b:free/17740754680",
    
]

for experiment_dir in experiments:
    experiment_dir = Path(experiment_dir)
    logger = logging.getLogger(__name__)
    with open(experiment_dir / "config.json", "r") as f:
        cfg = json.load(f)

    try:
        encoder_factory = EncoderFactory(cfg, experiment_dir, logger)
        encoder_factory.execute()
    except Exception as e:
        print(f"Error encoding embeddings: {e}")
        continue

    try:
        metrics_factory = MetricsFactory(cfg, experiment_dir, logger)
        metrics_factory.execute()
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        continue
