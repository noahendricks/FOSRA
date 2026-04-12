"""
file routes: list files, read file content, git status, grep, glob, find symbols.

uses deepagents FilesystemBackend for file operations.
grep falls back to ripgrep then python regex.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend.src.api.schemas.tui_schemas import (
    PROJECT_DIR,
    FileNode,
)

router = APIRouter(prefix="/oc", tags=["File"])


def _get_filesystem_backend():
    from deepagents.backends import FilesystemBackend

    return FilesystemBackend(root_dir=PROJECT_DIR)


def _file_info_to_node(info: Any) -> FileNode:  # type: ignore[reportExplicitAny]
    return FileNode(
        name=os.path.basename(info.path),  # type: ignore[reportUnknownMemberType]
        path=info.path,  # type: ignore[reportUnknownMemberType]
        absolute=info.path,  # type: ignore[reportUnknownMemberType]
        type="directory" if info.is_dir else "file",  # type: ignore[reportUnknownMemberType]
        ignored=False,
    )


# @router.get("/file")
# async def list_files(
#     path: str = Query("", alias="path"),
# ):
#     """
#     list files in a directory (like ls).
#     returns list of FileNode objects.
#     """
#     backend = _get_filesystem_backend()
#     abs_path = os.path.join(PROJECT_DIR, path) if path else PROJECT_DIR
#
#     try:
#         infos = backend.ls_info(abs_path)
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail="Directory not found")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#     return [_file_info_to_node(info) for info in infos]
#
#
# @router.get("/file/content")
# async def read_file_content(
#     filePath: str = Query(..., alias="filePath"),
# ):
#     """
#     read file content. detects binary vs text.
#     supports offset and limit query params.
#     """
#     backend = _get_filesystem_backend()
#     joined = os.path.join(PROJECT_DIR, filePath)
#     abs_path = os.path.realpath(joined)
#
#     if not abs_path.startswith(os.path.realpath(PROJECT_DIR)):
#         raise HTTPException(status_code=403, detail="Path outside project directory")
#
#     try:
#         content = backend.read(abs_path, offset=0, limit=0)
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail="File not found")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#     if content.startswith("[TOOL_ERROR]") or not content:
#         raise HTTPException(status_code=500, detail=content)
#
#     is_binary = False
#     try:
#         _ = content.encode("utf-8")
#     except UnicodeEncodeError:
#         is_binary = True
#
#     mime_type, _ = mimetypes.guess_type(abs_path)
#
#     return FileContent(
#         type="binary" if is_binary else "text",
#         content=content,
#         encoding="base64" if is_binary else None,
#         mimeType=mime_type,
#     )
#
#
# @router.get("/file/status")
# async def git_file_status() -> list[FileDiff]:
#     """
#     git status --porcelain to show changed/added/deleted files.
#     returns list of FileDiff objects.
#     """
#     try:
#         stat_out = subprocess.check_output(
#             ["git", "diff", "--stat"],
#             cwd=PROJECT_DIR,
#             stderr=subprocess.DEVNULL,
#             text=True,
#         )
#     except Exception:
#         return []
#
#     diffs: list[FileDiff] = []
#     for line in stat_out.strip().split("\n"):
#         if not line:
#             continue
#         parts = line.split()
#         if len(parts) >= 2:
#             file = parts[-1]
#             diffs.append(
#                 FileDiff(
#                     file=file,
#                     before="",
#                     after="",
#                     additions=0,
#                     deletions=0,
#                     status="modified",
#                 )
#             )
#     return diffs
#
#
# @router.get("/find")
# async def grep_files(
#     pattern: str = Query(...),
#     path: str = Query(""),
#     glob: str = Query(""),
# ) -> list[Any]:
#     """
#     grep search via ripgrep with fallback to python regex.
#     returns matches with file, line, content.
#     """
#     backend = _get_filesystem_backend()
#     search_path = os.path.join(PROJECT_DIR, path) if path else PROJECT_DIR
#
#     matches = backend.grep_raw(pattern, search_path, glob or "*")
#     if isinstance(matches, str):
#         return []
#     return matches
#
#
# @router.get("/find/symbol")
# async def find_symbol(
#     pattern: str = Query(...),
#     path: str = Query(""),
# ) -> list[Any]:
#     """
#     simple regex search for symbol/function definitions.
#     searches for lines starting with common definition patterns.
#     """
#     backend = _get_filesystem_backend()
#     search_path = os.path.join(PROJECT_DIR, path) if path else PROJECT_DIR
#
#     definition_patterns = [
#         r"^def\s+\w+",
#         r"^class\s+\w+",
#         r"^async\s+def\s+\w+",
#         r"^const\s+\w+\s*=",
#         r"^let\s+\w+\s*=",
#         r"^function\s+\w+",
#         r"^interface\s+\w+",
#         r"^type\s+\w+\s*=",
#     ]
#     combined = "|".join(f"({p})" for p in definition_patterns)
#
#     try:
#         matches = backend.grep_raw(combined, search_path, "*")
#     except Exception:
#         return []
#
#     if isinstance(matches, str):
#         return []
#     return matches


@router.get("/find/file")
async def glob_files(
    pattern: str = Query(...),
    path: str = Query(""),
) -> list[FileNode]:
    """
    glob search for files matching a pattern.
    returns list of FileNode.
    """
    backend = _get_filesystem_backend()
    search_path = os.path.join(PROJECT_DIR, path) if path else PROJECT_DIR

    try:
        infos = backend.glob_info(pattern, search_path)  # type: ignore[reportUnknownMemberType]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [_file_info_to_node(info) for info in infos]  # type: ignore[return-value]


@router.get("/find/symbol")
async def find_symbol(
    pattern: str = Query(...),
    path: str = Query(""),
) -> list[Any]:  # type: ignore[reportExplicitAny,reportUnknownVariableType]
    """
    simple regex search for symbol/function definitions.
    searches for lines starting with common definition patterns.
    """
    backend = _get_filesystem_backend()
    search_path = os.path.join(PROJECT_DIR, path) if path else PROJECT_DIR

    definition_patterns = [
        r"^def\s+\w+",
        r"^class\s+\w+",
        r"^async\s+def\s+\w+",
        r"^const\s+\w+\s*=",
        r"^let\s+\w+\s*=",
        r"^function\s+\w+",
        r"^interface\s+\w+",
        r"^type\s+\w+\s*=",
    ]
    combined = "|".join(f"({p})" for p in definition_patterns)

    try:
        matches = backend.grep_raw(combined, search_path, "*")  # type: ignore[reportUnknownMemberType]
    except Exception:
        return []

    if isinstance(matches, str):
        return []
    return matches
