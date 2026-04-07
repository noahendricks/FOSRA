"""Query expansion service for the evolved retrieval pipeline.

Generates:
    - rewritten_query: A cleaned, deambiguated query for retrieval
    - checklist: 4-5 structured sub-questions covering all query nuance

Uses LLM tool calling / structured output for reliable JSON emission.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from ulid import ULID

from backend.src.domain.schemas.retrieval import ChecklistItem, QueryExpansion
from backend.src.services.session.utils.llm_utils import build_llm
from backend.src.services.session.utils.prompts import (
    QUERY_EXPANSION_SYSTEM_PROMPT,
    QUERY_EXPANSION_USER_TEMPLATE,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from backend.src.settings import LLMConfig


class QueryExpander:
    """Service for expanding queries into rewritten form + checklist."""

    @staticmethod
    async def expand(
        user_query: str,
        llm_config: LLMConfig,
        chat_history: str | None = None,
    ) -> QueryExpansion:
        """Expand a user query into rewritten form + checklist.

        Args:
            user_query: The original user query
            llm_config: LLM configuration for the expansion
            chat_history: Optional conversation history for context resolution

        Returns:
            QueryExpansion with rewritten_query and checklist
        """

        llm: BaseChatModel = build_llm(llm_config)

        history_str = chat_history if chat_history else "No conversation history."

        user_prompt = QUERY_EXPANSION_USER_TEMPLATE.format(
            user_query=user_query,
            chat_history=history_str,
        )

        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(content=QUERY_EXPANSION_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )

            raw = str(response.content).strip()

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)

            rewritten = parsed.get("rewritten_query", user_query)

            checklist_data = parsed.get("checklist", [])

            checklist = [
                ChecklistItem(
                    id=item.get("id") or str(ULID()),
                    question=item.get("question", ""),
                    answered=item.get("answered", False),
                )
                for i, item in enumerate(checklist_data)
            ]

            if not checklist:
                checklist = [
                    ChecklistItem(id=str(ULID()), question=user_query, answered=False)
                ]

            logger.debug(
                "Query expansion: {} items in checklist",
                len(checklist),
            )

            return QueryExpansion(rewritten_query=rewritten, checklist=checklist)

        except Exception as e:
            logger.warning("Query expansion failed, using fallback: {}", e)
            return QueryExpansion(
                rewritten_query=user_query,
                checklist=[
                    ChecklistItem(id=str(ULID()), question=user_query, answered=False)
                ],
            )
