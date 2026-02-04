from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, SecretStr, ConfigDict
from pydantic.v1.utils import to_camel


class _BaseModelFlex(BaseModel):
    """_BaseModelFlex with flexible config for attribute-based initialization."""

    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    model_config = _FLEXIBLE_CONFIG


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


class FileResponse(_BaseModelFlex):
    origin_path: str
    uploaded_at: datetime = Field(default_factory=datetime.now)
    name: str = ""
    hash: str = ""
