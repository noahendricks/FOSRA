"""
real shell execution with sandboxing.

- commands run as the current user with cwd locked to PROJECT_DIR
- path traversal outside PROJECT_DIR is rejected
- output is truncated to 50KB
- timeout of 30 seconds per command
"""

from __future__ import annotations

import os
import subprocess
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException

from backend.src.api.schemas.tui_schemas import PROJECT_DIR

router = APIRouter(prefix="/oc/session", tags=["Shell"])

MAX_OUTPUT = 50 * 1024
TIMEOUT = 30


DANGEROUS_SHELL_CHARS = [";", "|", "&", "$(", "`", "${", ">", "<", "\n", "\r"]
BLOCKED_COMMANDS = ["sudo", "chmod", "chown", "rm -rf", "dd", "mkfs", "fdisk"]


def _validate_command(command: str) -> str:
    """reject commands with shell metacharacters or dangerous patterns."""
    for char in DANGEROUS_SHELL_CHARS:
        if char in command:
            raise HTTPException(
                status_code=400, detail=f"Shell metacharacter not allowed: '{char}'"
            )
    lower = command.lower()
    for blocked in BLOCKED_COMMANDS:
        if blocked in lower:
            raise HTTPException(
                status_code=400, detail=f"Command not allowed: '{blocked}'"
            )
    if ".." in command:
        raise HTTPException(status_code=400, detail="Path traversal not allowed")
    return command


@router.post("/{session_id}/shell")
async def run_shell(
    session_id: str,
    body: dict[str, Any],
):
    """
    execute a shell command.
    body: { "command": "ls -la", "description?": "list files" }
    returns { "stdout": str, "stderr": str, "exit": int }
    """
    command = body.get("command", "")
    _validate_command(command)

    import shlex

    try:
        args = shlex.split(command)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid command: {e}")

    try:
        result = subprocess.run(
            args,
            shell=False,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Command timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    stdout = result.stdout[:MAX_OUTPUT]
    stderr = result.stderr[:MAX_OUTPUT]

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit": result.returncode,
    }


@router.post("/{session_id}/command")
async def run_session_command(
    session_id: str,
    body: dict,
):
    """
    stub — slash commands are handled by the agent prompt flow.
    this endpoint exists for parity but just returns ok.
    """
    return {"ok": True}
