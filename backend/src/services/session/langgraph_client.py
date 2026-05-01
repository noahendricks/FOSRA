"""LangGraph Studio client script for FOSRA.

Usage:
    python -m backend.src.services.session.langgraph_client
"""

from __future__ import annotations

import asyncio

from langgraph_sdk import get_client


async def main() -> None:
    client = get_client(url="http://localhost:2024")

    # List available assistants
    assistants = await client.assistants.search()
    print(f"Assistants: {assistants}")

    agent = assistants[0]

    # Create a thread
    thread = await client.threads.create()
    print(f"Thread: {thread}")

    # Stream a run
    input_ = {"messages": [{"role": "human", "content": "Hello, what can you do?"}]}

    print("Streaming...")
    async for chunk in client.runs.stream(
        thread["thread_id"],
        agent["assistant_id"],
        input=input_,
        stream_mode="messages",
    ):
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
