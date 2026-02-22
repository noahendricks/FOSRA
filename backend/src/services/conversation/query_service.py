from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.domain.exceptions import (
    QueryExpansionError,
    MetadataFilterError,
    FusionError,
)
from backend.src.services.conversation.llm_service import LLMService
from backend.src.storage.utils.converters import DomainStruct


# =============================================================================
# Query Expansion
# =============================================================================



