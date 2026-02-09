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
from backend.src.domain.schemas import UserPreferences
from backend.src.services.conversation.llm_service import LLMService
from backend.src.storage.utils.converters import DomainStruct

if TYPE_CHECKING:
    pass


# =============================================================================
# Query Expansion
# =============================================================================


#for use with langchain
@dataclass
class QueryExpander:
    enable_expansion: bool = True
    enable_hyde: bool = False
    enable_decomposition: bool = False
    max_sub_queries: int = 5

    async def process_query(
        self,
        query: str,
        session: AsyncSession,
        user_prefs: UserPreferences,
        strategy: str = "auto",
    ) -> dict[str, Any]:
        if not query.strip():
            return {"original_query": query, "queries": [query], "strategy": "none"}

        logger.debug(f"Processing query with strategy: {strategy}")

        result = {
            "original_query": query,
            "queries": [query],
            "strategy": strategy,
            "metadata": {},
        }

        try:
            if strategy == "auto":
                if self._is_complex_query(query):
                    strategy = "decomposition"
                elif self._is_open_ended(query):
                    strategy = "hyde"
                else:
                    strategy = "expansion"

            tasks = []
            if strategy in ("expansion", "multi"):
                tasks.append(self._expand_query(query, user_prefs=user_prefs))
            if strategy in ("hyde", "multi"):
                tasks.append(
                    self._generate_hypothetical_document(query, user_prefs=user_prefs)
                )
            if strategy in ("decomposition", "multi"):
                tasks.append(self._decompose_query(query, user_prefs=user_prefs))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in results:
                    if isinstance(r, Exception):
                        logger.warning(f"Query processing task failed: {r}")
                        continue
                    if isinstance(r, list):
                        result["queries"].extend(r)
                    elif isinstance(r, str) and r:
                        result["queries"].append(r)

            result["queries"] = list(dict.fromkeys(result["queries"]))[:15]

            logger.success(
                f"Query processing completed: {len(result['queries'])} queries generated"
            )
            return result

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise QueryExpansionError(
                query=query,
                strategy=strategy,
                reason=str(e),
            ) from e

    def _is_complex_query(self, query: str) -> bool:
        indicators = ["and", "or", "also", "compare", "difference", "vs"]
        return (
            any(ind in query.lower() for ind in indicators) or len(query.split()) > 15
        )

    def _is_open_ended(self, query: str) -> bool:
        open_words = ["what", "why", "how", "explain", "describe"]
        return any(query.lower().startswith(w) for w in open_words)

    async def _expand_query(
        self,
        query: str,
        user_prefs: UserPreferences,
    ) -> list[str]:
        logger.debug("Expanding query into alternative phrasings")

        try:
            llm: ChatLiteLLM = LLMService.get_llm_for_role(
                role="fast", user_prefs=user_prefs
            )

            prompt = (
                f"Generate 3 alternative phrasings for: '{query}'. "
                "Return only the queries, one per line."
            )

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = str(response.content)

            alternatives = [
                line.strip() for line in content.split("\n") if line.strip()
            ]

            logger.debug(f"Generated {len(alternatives)} alternative phrasings")
            return alternatives

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []

    async def _generate_hypothetical_document(
        self,
        query: str,
        user_prefs: UserPreferences,
    ) -> str:
        if not self.enable_hyde:
            return ""

        logger.debug("Generating hypothetical document (HyDE)")

        try:
            llm: ChatLiteLLM = LLMService.get_llm_for_role(
                role="fast", user_prefs=user_prefs
            )

            prompt = (
                f"Write a short, factual paragraph (3 sentences) that answers: "
                f"'{query}'."
            )

            response: AIMessage = await llm.ainvoke([HumanMessage(content=prompt)])
            hyde_doc = str(response.content)

            logger.debug("Generated HyDE document")
            return hyde_doc

        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return ""

    async def _decompose_query(
        self,
        query: str,
        user_prefs: UserPreferences,
    ) -> list[str]:
        if not self.enable_decomposition:
            return []

        logger.debug("Decomposing query into sub-questions")

        try:
            llm: ChatLiteLLM = LLMService.get_llm_for_role(
                role="fast", user_prefs=user_prefs
            )

            prompt: str = (
                f"Break this into 3 simpler sub-questions: '{query}'. "
                "Return one per line."
            )

            response: AIMessage = await llm.ainvoke([HumanMessage(content=prompt)])
            content = str(response.content)

            sub_questions: list[str] = [
                line.strip() for line in content.split("\n") if line.strip()
            ]

            logger.debug(f"Generated {len(sub_questions)} sub-questions")
            return sub_questions

        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            raise QueryExpansionError(
                query=query,
                strategy="decomposition",
                reason=str(e),
            ) from e


# =============================================================================
# Metadata Filtering
# =============================================================================


@dataclass
class FusionRetriever:
    k: int = 60

    def fuse_results(
        self,
        results_list: list[list[dict[str, Any]]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not results_list:
            return []

        if len(results_list) == 1:
            result = results_list[0][:top_k] if top_k else results_list[0]
            logger.debug("Single result list, no fusion needed")
            return result

        logger.debug(f"Fusing {len(results_list)} result lists with RRF (k={self.k})")

        try:
            rrf_scores: dict[str, float] = {}
            chunk_map: dict[str, dict[str, Any]] = {}

            for list_idx, result_list in enumerate(results_list):
                for rank, chunk in enumerate(result_list):
                    chunk_id = chunk.get("id", "")
                    if not chunk_id:
                        chunk_id = str(hash(chunk.get("content", "")))

                    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                        1.0 / (self.k + rank + 1)
                    )

                    if chunk_id not in chunk_map:
                        chunk_map[chunk_id] = chunk

            sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            fused: list[dict[str, Any]] = []
            for chunk_id, rrf_score in sorted_ids:
                chunk = dict(chunk_map[chunk_id])
                chunk["rrf_score"] = rrf_score
                fused.append(chunk)

            result = fused[:top_k] if top_k else fused

            logger.success(
                f"Fused results: {len(chunk_map)} unique chunks, "
                f"returning top {len(result)}"
            )
            return result

        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            raise FusionError(
                reason=str(e),
                result_lists_count=len(results_list),
            ) from e


# =============================================================================
# Query Service
# =============================================================================


class QueryService:
    async def reformulate_query_with_chat_history(
        user_query: str,
        user_prefs: UserPreferences,
        chat_history_str: str | None = None,
    ) -> str:
        if not user_query or not user_query.strip():
            return user_query

        logger.info(f"Reformulating query: '{user_query[:50]}...'")

        try:
            llm: ChatLiteLLM = LLMService.get_llm_for_role(
                role="logical", user_prefs=user_prefs
            )

            system_message = SystemMessage(
                content=f"""
                Today's date: {datetime.now().strftime("%Y-%m-%d")}
                You are a highly skilled AI assistant specializing in query optimization for advanced research.
                Your primary objective is to transform a user's initial query into a highly effective search query.
                This reformulated query will be used to retrieve information from diverse data sources.

                **Chat History Context:**
                {chat_history_str if chat_history_str else "No prior conversation history is available."}
                If chat history is provided, analyze it to understand the user's evolving information needs and the broader context of their request. Use this understanding to refine the current query, ensuring it builds upon or clarifies previous interactions.

                **Query Reformulation Guidelines:**
                Your reformulated query should:
                1.  **Enhance Specificity and Detail:** Add precision to narrow the search focus effectively, making the query less ambiguous and more targeted.
                2.  **Resolve Ambiguities:** Identify and clarify vague terms or phrases. If a term has multiple meanings, orient the query towards the most likely one given the context.
                3.  **Expand Key Concepts:** Incorporate relevant synonyms, related terms, and alternative phrasings for core concepts. This helps capture a wider range of relevant sources.
                4.  **Deconstruct Complex Questions:** If the original query is multifaceted, break it down into its core searchable components or rephrase it to address each aspect clearly. The final output must still be a single, coherent query string.
                5.  **Optimize for Comprehensiveness:** Ensure the query is structured to uncover all essential facets of the original request, aiming for thorough information retrieval suitable for research.
                6.  **Maintain User Intent:** The reformulated query must stay true to the original intent of the user's query. Do not introduce new topics or shift the focus significantly.

                **Crucial Constraints:**
                *   **Conciseness and Effectiveness:** While aiming for comprehensiveness, the reformulated query MUST be as concise as possible. Eliminate all unnecessary verbosity. Focus on essential keywords, entities, and concepts that directly contribute to effective retrieval.
                *   **Single, Direct Output:** Return ONLY the reformulated query itself. Do NOT include any explanations, introductory phrases (e.g., "Reformulated query:", "Here is the optimized query:"), or any other surrounding text or markdown formatting.

                Your output should be a single, optimized query string, ready for immediate use in a search system.
                """
            )

            human_message = HumanMessage(
                content=f"Reformulate this query for better research results: {user_query}"
            )

            response = await llm.agenerate(messages=[[system_message, human_message]])

            reformulated_query = response.generations[0][0].text.strip()

            if not reformulated_query:
                logger.warning("Reformulation returned empty, using original query")
                return user_query

            logger.success(f"Query reformulated: '{reformulated_query[:50]}...'")
            return reformulated_query

        except Exception as e:
            logger.error(f"Query reformulation failed: {e}, using original query")
            return user_query

    @staticmethod
    async def langchain_chat_history_to_str(
        chat_history: list[Any],
    ) -> str:
        chat_history_str = "\!chat_history!\n"

        for chat_message in chat_history:
            if isinstance(chat_message, HumanMessage):
                chat_history_str += f"<user>{chat_message.content}</user>\n"
            elif isinstance(chat_message, AIMessage):
                chat_history_str += f"<assistant>{chat_message.content}</assistant>\n"
            elif isinstance(chat_message, SystemMessage):
                chat_history_str += f"<system>{chat_message.content}</system>\n"

        chat_history_str += "!chat_history!"

        logger.debug(f"Converted {len(chat_history)} messages to chat history string")
        return chat_history_str
