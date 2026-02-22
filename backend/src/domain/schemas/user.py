from datetime import datetime

from msgspec import field

from backend.src.storage.utils.converters import DomainStruct, utc_now
class User(DomainStruct):
    """Base user properties."""

    user_id: str
    username: str
    created_at: datetime | None = None
    last_login: datetime | None = None


class UserLogin(DomainStruct):
    """Base user properties."""

    user_id: str
    username: str
    password: str
    enabled: bool = True


class UserUpdate(User):
    """Properties for updating a user."""

    # TODO: Needs to actually update fields; currently doesn't upadate *only* the fields necessary
    # WARN: Incomplete / Not Working
    name: str | None = None
    enabled: bool | None = None
    updated_at: datetime = field(default_factory=utc_now)
