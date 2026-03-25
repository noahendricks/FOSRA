"""
file routes.
"""

from __future__ import annotations

import os
from fastapi import APIRouter, Query

router = APIRouter(prefix="/file", tags=["File"])


@router.get("/find")
async def find_text(pattern: str = Query(...)):
    """Find text patterns using ripgrep."""
    return []


@router.get("/find/file")
async def find_files(
    query: str = Query(...),
    dirs: bool = Query(True),
    type: str | None = Query(None),
    limit: int = Query(10),
):
    """Find files by name/pattern."""
    return []


@router.get("/find/symbol")
async def find_symbols(query: str = Query(...)):
    """Find LSP workspace symbols."""
    return []


@router.get("/")
async def list_files(path: str = Query(...)):
    """List files/directories at path."""
    return []


@router.get("/content")
async def read_file(path: str = Query(...)):
    """Read file content."""
    return {}


@router.get("/status")
async def get_file_status():
    """Get git status of all files."""
    return []
