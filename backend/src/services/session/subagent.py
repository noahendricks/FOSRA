"""Subagent service for the agentic retrieval loop.

Each subagent iteration:
1. Receives: original query + current checklist + accumulated context
2. Assesses: which checklist items are answered by current context
3. Plans: targeted retrieval queries for uncovered items
4. Dies: no conversation history carried forward

Uses LLM tool calling / structured output for reliable JSON emission.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from ulid import ULID

from backend.src.domain.schemas.retrieval import (
    AccumulatedContext,
    ChecklistItem,
    RetrievalFilters,
    RetrievalQuery,
    RetrievalTarget,
    SubagentResult,
)
from backend.src.services.session.utils.llm_utils import build_llm
from backend.src.services.session.utils.prompts import (
    SUBAGENT_SYSTEM_PROMPT,
    SUBAGENT_USER_TEMPLATE,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from backend.src.settings import LLMConfig


class Subagent:
    """Single-iteration retrieval planning agent."""

    @staticmethod
    async def assess_and_plan(
        original_query: str,
        checklist: list[ChecklistItem],
        context: AccumulatedContext,
        iteration: int,
        llm_config: LLMConfig,
        max_iterations: int = 5,
    ) -> SubagentResult:
        """Run a single subagent iteration.

        Args:
            original_query: The user's original question
            checklist: Current checklist state
            context: Accumulated retrieval context
            iteration: Current loop iteration (1-based)
            llm_config: LLM configuration
            max_iterations: Maximum allowed iterations

        Returns:
            SubagentResult with updated checklist and retrieval queries
        """

        llm: BaseChatModel = build_llm(llm_config)

        # current checklist state
        checklist_json = json.dumps(
            [
                {"id": c.id, "question": c.question, "answered": c.answered}
                for c in checklist
            ],
            indent=2,
        )

        context_text = context.to_formatted_context()

        if not context_text:
            context_text = "No context retrieved yet."

        user_prompt = SUBAGENT_USER_TEMPLATE.format(
            original_query=original_query,
            checklist_json=checklist_json,
            context_text=context_text,
            iteration=iteration,
            max_iterations=max_iterations,
        )

        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(content=SUBAGENT_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )

            raw = str(response.content).strip()

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)

            updated_checklist = [
                ChecklistItem(
                    id=item.get("id") or str(ULID()),
                    question=item.get("question", ""),
                    answered=item.get("answered", False),
                )
                for i, item in enumerate(parsed.get("checklist", []))
            ]

            if not updated_checklist:
                updated_checklist = checklist

            all_answered = parsed.get("all_answered", False)

            retrieval_queries = []

            for q in parsed.get("retrieval_queries", []):
                filters = None
                if q.get("filters"):
                    filters = RetrievalFilters(
                        file_ids=q["filters"].get("file_ids"),
                        node_type=q["filters"].get("node_type"),
                        language=q["filters"].get("language"),
                    )

                target_str = q.get("target", "vector").lower()
                target = RetrievalTarget.VECTOR
                if target_str == "graph":
                    target = RetrievalTarget.GRAPH
                elif target_str == "both":
                    target = RetrievalTarget.BOTH

                retrieval_queries.append(
                    RetrievalQuery(
                        query=q.get("query", ""),
                        target=target,
                        filters=filters,
                    )
                )

            result = SubagentResult(
                checklist=updated_checklist,
                all_answered=all_answered,
                retrieval_queries=retrieval_queries,
            )

            answered_count = sum(1 for c in updated_checklist if c.answered)

            logger.debug(
                "Subagent iteration {}: {}/{} answered, {} queries planned",
                iteration,
                answered_count,
                len(updated_checklist),
                len(retrieval_queries),
            )

            return result

        except Exception as e:
            logger.warning("Subagent failed, using fallback: {}", e)

            unanswered = [c for c in checklist if not c.answered]
            fallback_queries = [
                RetrievalQuery(query=c.question, target=RetrievalTarget.VECTOR)
                for c in unanswered[:3]
            ]

            return SubagentResult(
                checklist=checklist,
                all_answered=False,
                retrieval_queries=fallback_queries,
            )
