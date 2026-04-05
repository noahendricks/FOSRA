"""shared pydantic base model with camelcase alias generation."""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class BaseModelFlex(BaseModel):
    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    model_config: ConfigDict = _FLEXIBLE_CONFIG  # pyright: ignore


class BaseModelFlexLower(BaseModelFlex):
    """variant with str_to_lower for case-insensitive string fields."""

    _FLEXIBLE_CONFIG_LOWER = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
        str_to_lower=True,
    )

    model_config: ConfigDict = _FLEXIBLE_CONFIG_LOWER  # pyright: ignore
