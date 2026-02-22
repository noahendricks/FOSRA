from typing import override
from msgspec import field
from backend.src.storage.utils.converters import DomainStruct


class Workspace(DomainStruct):
    """Base workspace properties."""

    user_id: str
    name: str = field(default="New Workspace")
    description: str | None = field(
        default=None,
    )
    workspace_id: str | None = None
    archived_convos: list[str] | None = None



class WorkspaceFull(Workspace):
    """Workspace with related entities loaded."""

    sources_count: int = 0
    conversations_count: int = 0

