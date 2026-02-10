import os
import asyncio
from pathlib import Path
from pyrit.setup import initialize_pyrit_async
from pyrit.memory import SQLiteMemory
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup.initializers import SimpleInitializer
from pyrit.executor.attack import PromptSendingAttack


EXPERIMENT_DIR = Path("./experiments/tests/")


async def test_initialization() -> None:
    # Create the database directory if it doesn't exist
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    assert EXPERIMENT_DIR.exists() and EXPERIMENT_DIR.is_dir()

    db_file = EXPERIMENT_DIR / "memory.db"
    await initialize_pyrit_async(
        memory_db_type="SQLite",
        db_path=str(db_file),
        initializers=[SimpleInitializer()],
    )
    
    assert db_file.exists()
    print("INITIALIZATION TEST: PASSED")


async def test_connectivity() -> str:
    target = OpenAIChatTarget()

    attack = PromptSendingAttack(objective_target=target)

    # Testing connectivity with a harmless objective
    result = await attack.execute_async(objective="Say hello in one sentence.")

    # Testing the memory database
    db_file = EXPERIMENT_DIR / "memory.db"
    assert db_file.exists()
    memory = SQLiteMemory(db_path=str(db_file))

    conversation = memory.get_conversation(conversation_id=result.conversation_id)
    
    assert conversation, "No messages found in the test conversation."

    print(f"Test conversation ID: {result.conversation_id}")
    for message in conversation:
        if message.role not in {"user", "assistant"}:
            continue
        values = message.get_values()
        joined = "\n".join(values)
        assert joined.strip(), f"Empty content for role '{message.role}'."
        print(f"{message.role}: {joined}")

    print("CONNECTIVITY TEST: PASSED")


async def main():
    await test_initialization()
    await test_connectivity()

if __name__ == "__main__":
    asyncio.run(main())
