import os
import asyncio
from pathlib import Path
from pyrit.setup import initialize_pyrit_async
from pyrit.memory import SQLiteMemory
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup.initializers import SimpleInitializer
from pyrit.executor.attack import PromptSendingAttack


DB_PATH = Path("./db")


async def test_initialization() -> None:
    # Create the database directory if it doesn't exist
    DB_PATH.mkdir(parents=True, exist_ok=True)
    assert DB_PATH.exists() and DB_PATH.is_dir()

    db_file = DB_PATH / "test.db"
    await initialize_pyrit_async(
        memory_db_type="SQLite",
        db_path=str(db_file),
        # TODO: add the initializer once OpenAI Chat API is added to .env
        # initializers=[SimpleInitializer()],
    )
    
    assert db_file.exists()
    print("INITIALIZATION TEST: PASSED")


async def test_connectivity() -> str:
    target = OpenAIChatTarget(
        # TODO: remove assignments once OpenAI Chat API is added to .env
        api_key=os.environ["OLLAMA_CHAT_KEY"],
        endpoint=os.environ["OLLAMA_CHAT_ENDPOINT"],
        model_name=os.environ["OLLAMA_CHAT_MODEL"],
    )

    attack = PromptSendingAttack(objective_target=target)

    # Testing connectivity with a harmless objective
    result = await attack.execute_async(objective="Say hello in one sentence.")

    # Testing the memory database
    db_file = DB_PATH / "test.db"
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
