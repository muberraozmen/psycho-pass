import pandas as pd
from pathlib import Path
from pyrit.memory.sqlite_memory import SQLiteMemory


def memory2parquet(memory_db_path: str, parquet_path: str):
    """
    Converts a PyRIT SQLite attack database into a Parquet file containing the conversation history and outcome.
    
    Args:
        memory_db_path (str): Path to the .db file (e.g., ./experiments/run_1/memory.db)
        parquet_path (str): Output path (e.g., ./experiments/run_1/dataset.parquet)
    """    
    # Path checks
    if not Path(memory_db_path).exists():
        raise FileNotFoundError(f"Database not found: {memory_db_path}")
        
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)    

    # 1. Connect to PyRIT SQLite DB
    memory = SQLiteMemory(db_path=str(memory_db_path))
    
    # 2. Fetch all attack results (High-level outcomes)
    attacks = memory.get_attack_results()
    print(f"Found {len(attacks)} attack records. Processing...")
    
    data = []
    
    for attack in attacks:
        # 3. Fetch full conversation history for each attack
        raw_msgs = memory.get_conversation(conversation_id=attack.conversation_id)
        
        # Sort by sequence to ensure chronological order (Turn 1, Turn 2...)
        raw_msgs.sort(key=lambda x: x.sequence)
        
        conversation_history = []
        
        for msg in raw_msgs:
            # Filter out system prompts for cleaner analysis
            if msg.role == "system":
                continue
            
            # Extract content 
            content = msg.get_values()[0] if msg.get_values() else ""
            
            # Build Message Object
            conversation_history.append({
                "sequence": msg.sequence,
                "role": msg.role,
                "content": content
            })
        
        # 4. Get outcome string
        try:
            # If PyRIT uses an Enum for outcome
            outcome_str = str(attack.outcome.value) 
        except AttributeError:
            # If it's already a string
            outcome_str = str(attack.outcome)

        # 5. Build the Data Row
        row = {
            "conversation_id": str(attack.conversation_id),
            "objective": attack.objective,
            "is_success": "success" in outcome_str,
            "is_failure": "failure" in outcome_str,
            "turn_count": len(conversation_history),
            "messages": conversation_history
        }
        data.append(row)

    df = pd.DataFrame(data)
    
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    print(f"Saved {len(df)} rows as {parquet_path}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_db_path", type=str, required=True)
    parser.add_argument("--parquet_path", type=str, required=True)
    args = parser.parse_args()
    memory2parquet(args.memory_db_path, args.parquet_path)

# python src/processors.py --memory_db_path /Users/muberraozmen/Development/psycho-pass/experiments/dataset_feb3_1770230537/memory.db --parquet_path /Users/muberraozmen/Development/psycho-pass/experiments/dataset_feb3_1770230537/dataset.parquet
