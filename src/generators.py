from pyrit.setup import initialize_pyrit_async
from pyrit.datasets import SeedDatasetProvider
from src.attacks import RTA


async def attack_generator(
    db_path: str,
    cfg: dict,
    ) -> None:

    # Step 1: Initialize the database
    memory = await initialize_pyrit_async(
        memory_db_type="SQLite", db_path=db_path
        )

    # Step 2: Fetch the seed dataset 
    seed_name = cfg.get("seed", {}).get("name", "adv_bench")
    seed_dataset = await SeedDatasetProvider.fetch_datasets_async(
        dataset_names=[seed_name]
    )
    seeds = seed_dataset[0].seeds
    num_samples = cfg.get("seed", {}).get("num_samples", 3)

    # Step 3: Make the attack
    attack_name = cfg.get("attack", {}).get("name", "rta")
    if attack_name == "rta":
        attack = RTA(cfg["attack"])
    else:
        raise ValueError(f"Unsupported attack name: {attack_name}")

    # Step 4: Run the attacks
    total_num_attacks = num_samples * len(seeds)
    print(f"Total number of attacks to generate: {total_num_attacks}")
    print(f"Destination: {db_path}")
    counter = 0
    for seed in seeds:
        for _ in range(num_samples):
            await attack.execute_async(
                objective=seed.value, 
                memory_labels={
                    "data_type": seed.data_type if seed.data_type else "text",
                    "harm_categories": seed.harm_categories if seed.harm_categories else None,
                }
            )
            print(f"Attack {counter + 1} of {total_num_attacks} completed")
            counter += 1
            
    # TODO: add the parallel execution logic based on together.ai's cookbook
    # https://github.com/togethercomputer/together-cookbook/blob/main/Agents/Parallel_Agent_Workflow.ipynb
    
