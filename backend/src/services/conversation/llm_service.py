from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from loguru import logger

from backend.src.settings import (
    LLMConfig,
    ScoredRetrieval,
    UserPreferences,
)
from backend.src.domain.schemas.doc import Chunk, Doc
from backend.src.services.conversation.utils.llm_utils import (
    build_llm,
    format_sources_section,
    langchain_chat_history_to_str,
    ui_messages_to_lc_messages,
)
from backend.src.services.conversation.utils.prompts import (
    DOC_TOPIC_GEN_PROMPT,
    FOSRA_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from langchain_community.chat_models import ChatLiteLLM

    from backend.src.api.schemas.api_schemas import UIMessage


class LLMService:
    """Thin wrapper around LiteLLM for FOSRA's generation needs.

    Every method accepts an ``LLMConfig`` (or extracts one from
    ``UserPreferences``) so callers control which model is used.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_llm_config(user_prefs: UserPreferences | None) -> LLMConfig:
        """Pick the best available LLM config from user preferences.

        Precedence: llm_default > llm_logic > llm_fast > llm_heavy > fallback.
        """
        if user_prefs:
            for cfg in (
                user_prefs.llm_default,
                user_prefs.llm_logic,
                user_prefs.llm_fast,
                user_prefs.llm_heavy,
            ):
                if cfg is not None:
                    return cfg
        # Last-resort fallback (will need a valid API key at runtime)
        return LLMConfig()

    # ------------------------------------------------------------------
    # 1. Chat Response Generation (streaming)
    # ------------------------------------------------------------------

    @staticmethod
    async def generate_llm_response(
        chat_history: list[UIMessage],
        sources: list[ScoredRetrieval],
        convo_id: str,
        user_prefs: UserPreferences | None,
    ) -> AsyncIterator:
        """Stream an LLM response given chat history and retrieved sources.

        Uses the user's configured LLM from ``user_prefs``.  Injects the
        FOSRA system prompt and citation instructions around the retrieved
        ``sources``.
        """
        config = LLMService._resolve_llm_config(user_prefs)
        llm: ChatLiteLLM = build_llm(config)

        # Convert UI messages to LangChain format
        lc_messages: list[BaseMessage] = ui_messages_to_lc_messages(
            ui_messages=chat_history
        )

        if not lc_messages:
            raise ValueError("No messages provided to generate response")

        # The last user message is the current question
        newest_message: BaseMessage = lc_messages.pop()

        source_content: str = format_sources_section(sources)

        instruction_text = (
            "Please provide a detailed, comprehensive answer to the user's "
            "question using the information from their personal knowledge "
            "sources. Make sure to cite all information appropriately and "
            "engage in a conversational manner."
        )

        human_message_content = (
            f"{source_content}\n\n"
            f"User's question:\n"
            f"<user_query>\n{newest_message.content}\n</user_query>\n\n"
            f"{instruction_text}"
        )

        # Build final message list: system → history → current turn
        final_messages: list[BaseMessage] = [
            SystemMessage(content=FOSRA_SYSTEM_PROMPT),
            *lc_messages,  # prior conversation turns (minus the latest)
            HumanMessage(content=human_message_content),
        ]

        logger.debug(
            "Generating LLM response for convo={} with {} sources",
            convo_id,
            len(sources),
        )
        return llm.astream(input=final_messages)

    # ------------------------------------------------------------------
    # 2. Document Topic Classification
    # ------------------------------------------------------------------

    @staticmethod
    async def classify_chunk_topic(
        chunk_text: str,
        existing_topics: list[str],
        *,
        llm_config: LLMConfig,
    ) -> AsyncIterator:
        """Classify a single chunk into a topic using the LLM.

        Returns a streaming iterator (caller should collect and parse JSON).
        """
        llm: ChatLiteLLM = build_llm(llm_config)
        prompt = DOC_TOPIC_GEN_PROMPT.format(
            existing_topics=existing_topics, chunk_text=chunk_text
        )
        return llm.astream(input=prompt)

    @staticmethod
    async def generate_chunk_summaries(
        parent_chunks: list[Chunk],
        existing_topics: list[str] | None = None,
        *,
        llm_config: LLMConfig,
    ) -> list[dict[str, Any]]:
        """Classify a batch of parent chunks into topics.

        TODO: Implement batched classification (3 chunks at a time),
        accumulate, then consolidate with a second LLM pass.
        """
        # Placeholder — will be implemented when topic pipeline is wired up
        logger.warning("generate_chunk_summaries is not yet implemented")
        return []

    # ------------------------------------------------------------------
    # 3. Graph stubs (Phase B / C)
    # ------------------------------------------------------------------

    @staticmethod
    async def generate_graph_query(query: str, *, llm_config: LLMConfig) -> str | None:
        """Generate a Cypher query for the code knowledge graph.

        TODO: Phase B implementation.
        """
        logger.warning("generate_graph_query is not yet implemented")
        return None

    @staticmethod
    async def generate_graph_search(
        query: str, *, llm_config: LLMConfig
    ) -> dict[str, Any] | None:
        """Generate vector search parameters for the code graph.

        TODO: Phase B implementation.
        """
        logger.warning("generate_graph_search is not yet implemented")
        return None

    # ------------------------------------------------------------------
    # 4. Document summarization stub
    # ------------------------------------------------------------------

    @staticmethod
    async def summarize_document(text: str, *, llm_config: LLMConfig) -> str | None:
        """Produce a concise summary of a full document.

        TODO: Implement when document-level metadata pipeline is built.
        """
        logger.warning("summarize_document is not yet implemented")
        return None
