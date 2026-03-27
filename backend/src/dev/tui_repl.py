# %% Cell 1 - Imports
import asyncio
import json
import time
from pathlib import Path

import httpx
from icecream import ic
from rich.console import Console

# Rich and Icecream for pretty printing
from rich.pretty import pprint as pp
from rich.table import Table
from rich.traceback import install

install(show_locals=True)

console = Console()
ic.configureOutput(prefix="DEBUG | ", includeContext=True)

BASE_URL = "http://localhost:8000"
pp("All imports successful!")

# %% Cell 2 - Start the App
# Run: uvicorn backend.src.app:app --reload --port 8000
# Then execute remaining cells


# %% Cell 3 - Health & Version
async def check_health():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/oc/version")
        return r.json()


health = await check_health()

pp("=== VERSION ===")
ic(health)


# %% Cell 4 - List Skills
async def list_skills():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/oc/skill")
        return r.json()


# NOTE: Not working
skills = await list_skills()  # pyright: ignore
pp("=== SKILLS ===")
ic(skills)


# %% Cell 5 - Session CRUD
DEFAULT_USER_ID = "dev-user"


async def create_session():
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE_URL}/oc/session",
            json={"user_id": DEFAULT_USER_ID},
        )
        return r.json()


async def list_sessions():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/oc/session")
        return r.json()


session = await create_session()  # pyright: ignore
pp("=== SESSION CREATED ===")
ic(session)

sessions = await list_sessions()  # pyright: ignore
pp("=== SESSIONS ===")
ic(sessions)


# %% Cell 6 - Publish a TuiEvent and Subscribe to SSE
SESSION_ID = session.get("sessionID", "")


async def publish_test_event():
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE_URL}/oc/session/{SESSION_ID}/message",
            json={
                "content": [{"part": "text", "text": "hello world"}],
                "stream": True,
            },
        )
        return r.json()


async def subscribe_sse(duration: float = 5.0):
    """Subscribe to SSE stream and collect events for `duration` seconds."""
    events = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(duration + 5.0)) as client:
        async with client.stream("GET", f"{BASE_URL}/oc/event") as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                if line.startswith("data: "):
                    data = line[len("data: ") :]
                    try:
                        events.append(json.loads(data))
                    except Exception:
                        pass
                elif line.startswith("event: "):
                    event_type = line[len("event: ") :].strip()
                    if events:
                        events[-1]["_event"] = event_type
                elif line.startswith("id: "):
                    event_id = line[len("id: ") :].strip()
                    if events:
                        events[-1]["_id"] = event_id

    return events


async def test_sse_with_event():
    """Publish a message then immediately subscribe to SSE to capture the event."""
    import asyncio

    sse_task = asyncio.create_task(subscribe_sse(duration=3.0))
    await asyncio.sleep(0.5)
    await publish_test_event()
    events = await sse_task
    return events


pp("=== SUBSCRIBING TO SSE ===")
events = await test_sse_with_event()  # pyright: ignore
pp(f"=== SSE EVENTS ({len(events)}) ===")

table = Table(title="SSE Events")
table.add_column("ID", style="cyan")
table.add_column("Type", style="green")
table.add_column("Preview", style="white")

for ev in events:
    ev_type = ev.get("type", "unknown")
    props = ev.get("properties", {})
    preview = json.dumps(props)[:80]
    ev_id = ev.get("_id", "-")
    table.add_row(ev_id, ev_type, preview)

console.print(table)


# %% Cell 7 - Path Aliases
async def check_aliases():
    async with httpx.AsyncClient() as client:
        lsp = await client.get(f"{BASE_URL}/oc/lsp")
        mcp = await client.get(f"{BASE_URL}/oc/mcp")
        formatter = await client.get(f"{BASE_URL}/oc/formatter")
        return {
            "lsp": (lsp.status_code, lsp.json()),
            "mcp": (mcp.status_code, mcp.json()),
            "formatter": (formatter.status_code, formatter.json()),
        }


aliases = await check_aliases()  # pyright: ignore
pp("=== PATH ALIASES ===")
ic(aliases)


# %% Cell 8 - Get Session Messages
async def get_messages(sid: str):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/oc/session/{sid}/message")
        return r.json()


if SESSION_ID:
    msgs = await get_messages(SESSION_ID)  # pyright: ignore
    pp("=== SESSION MESSAGES ===")
    ic(len(msgs))
    for msg in msgs[:3]:
        ic(msg.get("messageId"), msg.get("role"))


# %% Cell 9 - Stress Test: Many Events
async def stress_test():
    """Publish 20 events rapidly and verify none dropped via SSE replay."""
    import asyncio

    async def publish_many():
        async with httpx.AsyncClient() as client:
            for i in range(20):
                await client.post(
                    f"{BASE_URL}/oc/session/{SESSION_ID}/message",
                    json={
                        "content": [{"part": "text", "text": f"stress test {i}"}],
                        "stream": True,
                    },
                )
                await asyncio.sleep(0.01)

    sse_task = asyncio.create_task(subscribe_sse(duration=3.0))
    await asyncio.sleep(0.2)
    await publish_many()
    events = await sse_task

    return {
        "published": 20,
        "received": len(events),
        "types": [e.get("type") for e in events],
    }


stress = await stress_test()  # pyright: ignore
pp("=== STRESS TEST ===")
ic(stress)


# %% Cell 10 - Shutdown Event
# To test: kill the server process
# You should see "server.instance.disposed" in the SSE stream before the connection closes


pp("=== REPL Ready for Interactive Testing ===")
pp(
    {
        "available_variables": [
            "health",
            "skills",
            "session",
            "sessions",
            "events",
            "aliases",
            "msgs",
            "stress",
        ]
    }
)
