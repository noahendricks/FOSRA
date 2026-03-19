from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from backend.src.services.conversation.utils.llm_utils import (
    build_llm,
    langchain_chat_history_to_str,
)
from backend.src.services.conversation.utils.prompts import (
    COVERAGE_CHECK_PROMPT_TEMPLATE,
    QUERY_REFORM_PROMPT,
    SPLIT_SUBQUERIES_PROMPT_TEMPLATE,
)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from backend.src.domain.schemas.config import LLMConfig


# ------------------------------------------------------------------
# Return types
# ------------------------------------------------------------------


@dataclass
class CoverageResult:
    """Result of a sub-query coverage check."""

    coverage: dict[str, bool] = field(default_factory=dict)
    covered_count: int = 0
    total_count: int = 0

    @property
    def fully_covered(self) -> bool:
        return self.covered_count >= self.total_count

    @property
    def uncovered_queries(self) -> list[str]:
        return [q for q, covered in self.coverage.items() if not covered]


# ------------------------------------------------------------------
# Service
# ------------------------------------------------------------------


class QueryService:
    """Stateless helpers for query reformulation, decomposition, and
    coverage assessment.  Each method accepts an ``LLMConfig`` so the
    caller decides which model to use (fast / logic / heavy).
    """

    # ------------------------------------------------------------------
    # 1. Query Reformulation
    # ------------------------------------------------------------------

    @staticmethod
    async def reform_query(
        user_query: str,
        chat_history: list[BaseMessage] | None = None,
        existing_topics: list[str] | None = None,
        *,
        llm_config: LLMConfig,
    ) -> str:
        """Reformulate *user_query* using conversation context.

        Returns a single self-contained retrieval query string.
        If the LLM call fails, the original query is returned unchanged.
        """
        history_str = (
            langchain_chat_history_to_str(chat_history) if chat_history else "[]"
        )
        topics_str = ", ".join(existing_topics) if existing_topics else "[]"

        prompt = (
            QUERY_REFORM_PROMPT.replace("{{existing_topics}}", topics_str)
            .replace("{{conversation_history}}", history_str)
            .replace("{{user_query}}", user_query)
        )

        llm = build_llm(llm_config)

        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(
                        content="You are a query reformulation engine. Output ONLY the reformulated query string."
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            reformed = str(response.content).strip()
            if not reformed:
                logger.warning(
                    "LLM returned empty reform — falling back to original query"
                )
                return user_query
            logger.debug("Reformulated query: {}", reformed)
            return reformed
        except Exception:
            logger.exception("reform_query failed — returning original query")
            return user_query

    # ------------------------------------------------------------------
    # 2. Sub-query Decomposition
    # ------------------------------------------------------------------

    @staticmethod
    async def split_to_subqueries(
        user_query: str,
        *,
        llm_config: LLMConfig,
    ) -> list[str]:
        """Decompose *user_query* into atomic sub-queries for coverage
        assessment.

        Returns a list of 1-6 sub-query strings.  On failure returns a
        single-element list containing the original query.
        """
        # SPLIT_SUBQUERIES_PROMPT_TEMPLATE uses {user_query} (single-brace)
        prompt = SPLIT_SUBQUERIES_PROMPT_TEMPLATE.format(user_query=user_query)

        llm = build_llm(llm_config)

        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(
                        content="You are a query decomposition engine. Output ONLY a JSON array of strings."
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            raw = str(response.content).strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            sub_queries: list[str] = json.loads(raw)
            if not isinstance(sub_queries, list) or not sub_queries:
                raise ValueError(f"Expected non-empty JSON array, got: {raw!r}")
            logger.debug("Split into {} sub-queries", len(sub_queries))
            return [str(q) for q in sub_queries]
        except Exception:
            logger.exception(
                "split_to_subqueries failed — returning original query as single sub-query"
            )
            return [user_query]

    # ------------------------------------------------------------------
    # 3. Coverage Check
    # ------------------------------------------------------------------

    @staticmethod
    async def check_coverage(
        sub_queries: list[str],
        context: str,
        *,
        llm_config: LLMConfig,
    ) -> CoverageResult:
        """Check which *sub_queries* are answered by the retrieved *context*.

        Returns a ``CoverageResult`` with per-query coverage booleans.
        On failure returns everything as uncovered so retrieval can retry.
        """
        sub_queries_str = json.dumps(sub_queries, indent=2)

        # COVERAGE_CHECK_PROMPT_TEMPLATE uses {{var}} double-braces
        # (not compatible with PromptTemplate — do manual replacement)
        prompt = COVERAGE_CHECK_PROMPT_TEMPLATE.replace(
            "{{sub_queries}}", sub_queries_str
        ).replace("{{context}}", context)

        llm = build_llm(llm_config)

        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(
                        content="You are a retrieval coverage assessor. Output ONLY a JSON object."
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            raw = str(response.content).strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed: dict = json.loads(raw)

            coverage_map: dict[str, bool] = parsed.get("coverage", {})
            covered_count = parsed.get(
                "covered_count", sum(1 for v in coverage_map.values() if v)
            )
            total_count = parsed.get("total_count", len(sub_queries))

            result = CoverageResult(
                coverage=coverage_map,
                covered_count=covered_count,
                total_count=total_count,
            )
            logger.debug("Coverage: {}/{}", result.covered_count, result.total_count)
            return result
        except Exception:
            logger.exception("check_coverage failed — treating all as uncovered")
            return CoverageResult(
                coverage={q: False for q in sub_queries},
                covered_count=0,
                total_count=len(sub_queries),
            )
