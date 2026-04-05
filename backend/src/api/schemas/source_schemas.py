"""source schemas — FileNode, FileContent, File, VcsInfo, LspStatus, FormatterStatus, MCP, ToolList, etc."""

from __future__ import annotations

from typing import Any, Literal

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
    lines: list[str]


class FileContentPatch(BaseModel):
    oldFileName: str
    newFileName: str
    oldHeader: str | None = None
    newHeader: str | None = None
    hunks: list[FileContentPatchHunk]
    index: str | None = None


class FileContent(BaseModel):
    type: Literal["text", "binary"]
    content: str
    diff: str | None = None
    patch: FileContentPatch | None = None
    encoding: Literal["base64"] | None = None
    mimeType: str | None = None


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
    extensions: list[str]
    enabled: bool


# ---- MCP ----


class McpResource(BaseModel):
    name: str
    uri: str
    description: str | None = None
    mimeType: str | None = None
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
