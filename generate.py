import argparse
import json
from pathlib import Path
from datetime import datetime
import logging
import asyncio

from src.generators import attack_generator


logging.basicConfig(level=logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_root", type=str, default="./db")
    parser.add_argument("--config_path", type=str, default="./configs/rta_base.json")

    args = parser.parse_args()

    
    with open(args.config_path, "r") as f:
        cfg = json.load(f)

    # Make the run directory and save the config
    db_root = args.db_root + "/" + f"{cfg['attack']['name']}_{cfg['seed']['name']}_{int(datetime.now().timestamp())}"
    Path(db_root).mkdir(parents=True, exist_ok=True)
    with open(db_root + "/config.json", "w") as f:
        json.dump(cfg, f)

    # Generate the attacks
    asyncio.run(attack_generator(
        cfg=cfg,
        db_path=db_root + "/attacks.db",
    ))

