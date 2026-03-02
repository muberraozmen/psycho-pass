from pathlib import Path
import sqlite3
import pandas as pd


def extract_messages(db_path: Path | str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        attack_result_entries = pd.read_sql_query("SELECT * FROM AttackResultEntries;", conn)
        prompt_memory_entries = pd.read_sql_query("SELECT * FROM PromptMemoryEntries;", conn)

    df = prompt_memory_entries.merge(
        attack_result_entries,
        on=["conversation_id", "attack_identifier"],
        how="left",
        suffixes=("", "_attack_result"),
    )

    df.rename(columns={
        "id_attack_result": "attack_result_id", 
        "original_value": "value"}, 
        inplace=True
    )

    df = df[~df["attack_result_id"].isna()]

    columns = [
        "attack_result_id", "conversation_id", "last_score_id",
        "objective", "executed_turns", "outcome", "outcome_reason",
        "id", "role", "sequence", "value"
    ]
    df = df[columns]

    df.sort_values(by=["attack_result_id", "sequence"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
