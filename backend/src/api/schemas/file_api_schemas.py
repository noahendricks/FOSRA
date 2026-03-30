from datetime import datetime
from typing import Any

from backend.src.api.schemas.base import _BaseModelFlex


class FileRequest(_BaseModelFlex):
    origin_path: str


class FileResponse(_BaseModelFlex):
    origin_path: str
    file_type: str
    size: int
    last_modified: datetime | None = None
    content_type: str | None = None
    metadata: dict[str, Any] = {}

    class Config:
        from_attributes = True
