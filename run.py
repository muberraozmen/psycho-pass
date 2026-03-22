import argparse
import asyncio
from dotenv import load_dotenv

load_dotenv(".env")

from src.utils import make_experiment
from src.factory import AttackFactory, EncoderFactory, MetricsFactory




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="Name of the experiment run")
    parser.add_argument("--config_file", type=str, default="base.yaml", help="Base config file")
    parser.add_argument("--max_concurrency", type=int, default=3, help="Max concurrent attacks")
    
    args = parser.parse_args()
    
    cfg, experiment_dir, logger = make_experiment(args)

    attack_factory = AttackFactory(cfg, experiment_dir, logger)
    encoder_factory = EncoderFactory(cfg, experiment_dir, logger)
    metrics_factory = MetricsFactory(cfg, experiment_dir, logger)

    asyncio.run(attack_factory.execute())
    encoder_factory.execute()
    metrics_factory.execute()


if __name__ == "__main__":
    main()














