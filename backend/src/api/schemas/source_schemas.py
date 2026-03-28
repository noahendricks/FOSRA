"""source schemas — FileNode, FileContent, File, VcsInfo, LspStatus, FormatterStatus, MCP, ToolList, etc."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel


class FileNode(BaseModel):
    name: str
    path: str
    absolute: str
    type: Literal["file", "directory"]
    ignored: bool


class FileContentPatchHunk(BaseModel):
    oldStart: int
    oldLines: int
    newStart: int
    newLines: int
    lines: List[str]


class FileContentPatch(BaseModel):
    oldFileName: str
    newFileName: str
    oldHeader: Optional[str] = None
    newHeader: Optional[str] = None
    hunks: List[FileContentPatchHunk]
    index: Optional[str] = None


class FileContent(BaseModel):
    type: Literal["text", "binary"]
    content: str
    diff: Optional[str] = None
    patch: Optional[FileContentPatch] = None
    encoding: Optional[Literal["base64"]] = None
    mimeType: Optional[str] = None


class File(BaseModel):
    path: str
    added: int
    removed: int
    status: Literal["added", "deleted", "modified"]


class VcsInfo(BaseModel):
    branch: str


# ---- LSP / FORMATTER ----


class LspStatus(BaseModel):
    id: str
    name: str
    root: str
    status: Literal["connected", "error"]


class FormatterStatus(BaseModel):
    name: str
    extensions: List[str]
    enabled: bool


# ---- MCP ----


class McpResource(BaseModel):
    name: str
    uri: str
    description: Optional[str] = None
    mimeType: Optional[str] = None
    client: str


McpStatus = Any  # Union of status variants — avoid circular


class ToolIds(BaseModel):
    pass


class ToolListItem(BaseModel):
    id: str
    description: str
    parameters: Any


class ToolList(BaseModel):
    pass
