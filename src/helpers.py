
def describe_db(db_path: str) -> None:
    import sqlite3
    from pathlib import Path

    def _get_tables(conn: sqlite3.Connection) -> list[str]:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        return [row[0] for row in cursor.fetchall()]

    def _get_columns(conn: sqlite3.Connection, table_name: str) -> list[dict[str, str]]:
        cursor = conn.execute(f"PRAGMA table_info({table_name});")
        columns = []
        for cid, name, col_type, notnull, default_value, pk in cursor.fetchall():
            columns.append(
                {
                    "cid": str(cid),
                    "name": name,
                    "type": col_type,
                    "notnull": str(notnull),
                    "default": str(default_value) if default_value is not None else "",
                    "pk": str(pk),
                }
            )
        return columns

    def _get_row_count(conn: sqlite3.Connection, table_name: str) -> int:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name};")
        return int(cursor.fetchone()[0])

    def _describe_table(conn: sqlite3.Connection, table_name: str) -> None:
        columns = _get_columns(conn, table_name)
        row_count = _get_row_count(conn, table_name)
        print(f"\nTable: {table_name}")
        print(f"Rows: {row_count}")
        print("Columns:")
        for column in columns:
            details = (
                f"  - {column['name']} ({column['type']})"
                f" | pk={column['pk']} | notnull={column['notnull']}"
            )
            if column["default"]:
                details += f" | default={column['default']}"
            print(details)

        if row_count == 0:
            print("Samples: (no rows)")
            return

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        tables = _get_tables(conn)
        print(f"Database: {db_path}")
        print(f"Tables: {len(tables)}")
        for table_name in tables:
            _describe_table(conn, table_name)


def view_conversations(db_path: str, n_conversations: int = 10) -> None:
    from pathlib import Path
    from pyrit.memory.sqlite_memory import SQLiteMemory

    memory = SQLiteMemory(db_path=Path(db_path))

    attacks = memory.get_attack_results()

    for attack in attacks[:n_conversations]: 
        print("\n================================================")
        print(f"Conversation ID: \t {attack.conversation_id}")
        print(f"Result: \t\t {attack.outcome}")
        conversation = memory.get_conversation(conversation_id=attack.conversation_id)
        for message in conversation:
            print("------------------------------------------------")
            print(f"TURN {message.sequence} - ROLE {message.role}")
            print(f"{message.get_values()[0]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--n_conversations", type=int, default=3)
    args = parser.parse_args()
    describe_db(args.db_path)
    view_conversations(args.db_path, args.n_conversations)