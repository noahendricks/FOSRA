from __future__ import annotations

from typing import TYPE_CHECKING, Any

import litellm
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_litellm import ChatLiteLLM
from loguru import logger

from backend.src.api.schemas.api_schemas import UIMessage
from backend.src.domain.schemas.config import (
    LLMConfig,
    ScoredRetrieval,
    UserPreferences,
)
from backend.src.domain.schemas.doc import Chunk, Doc
from backend.src.services.conversation.utils.llm_utils import (
    build_model_string,
    filter_none_values,
    format_sources_section,
    langchain_chat_history_to_str,
    ui_messages_to_lc_messages,
)
from backend.src.services.conversation.utils.prompts import FOSRA_SYSTEM_PROMPT

if TYPE_CHECKING:
    pass

litellm.drop_params = True


PROVIDER_TO_LITELLM_MAP: dict[str, str] = {
    "OPENAI": "openai",
    "ANTHROPIC": "anthropic",
    "COHERE": "cohere",
    "GROQ": "groq",
    "TOGETHER": "together_ai",
    "MISTRAL": "mistral",
    "REPLICATE": "replicate",
    "HUGGINGFACE": "huggingface",
    "BEDROCK": "bedrock",
    "VERTEX_AI": "vertex_ai",
    "PALM": "palm",
    "OPENROUTER": "openrouter",
}


from backend.src.services.conversation.utils.prompts import (
    COVERAGE_CHECK_PROMPT,
    DOC_TOPIC_GEN_PROMPT,
    FOSRA_CITATION_INSTRUCTIONS,
    GRAPH_QUERY_GEN_PROMPT,
    GRAPH_SEARCH_PROMPT,
    SPLIT_SUBQUERIES_PROMPT,
)


class LLMService:

    @staticmethod
    async def generate_llm_response(
        chat_history: list[UIMessage],
        sources: list[ScoredRetrieval],
        convo_id: str,
        user_prefs: UserPreferences | None,
    ):
        lc_messages: list[BaseMessage] = ui_messages_to_lc_messages(
            ui_messages=chat_history
        )

        source_content: str = format_sources_section(sources)

        # if not lc_messages:
        #     raise ValueError("No messages provided to generate response")

        # newest_message: BaseMessage = lc_messages[-1]
        newest_message: BaseMessage = HumanMessage(
            content="what are is evaluator optimizer?"
        )

        config: LLMConfig = LLMConfig(
            config_id=0,
            config_name="the config name",
            provider="ollama",
            model="qwen2.5:7b",
            api_key="dummy",
            api_base="http://localhost:11434",
        )

        model_string: str = build_model_string(
            provider=config.provider,
            model_name=config.model,
            custom_provider=config.custom_provider,
        )

        kwargs: dict[str, Any] = {
            "model": model_string,
            "api_key": config.api_key,
            "streaming": True,
        }

        if config.api_base:
            print("DEBUGPRINT[87]: llm_service.py:94 (after if config.api_base:)")
            kwargs["api_base"] = config.api_base

        if config.litellm_params:
            print("DEBUGPRINT[88]: llm_service.py:98 (after if config.litellm_params:)")
            kwargs.update(config.litellm_params)

        llm: ChatLiteLLM = ChatLiteLLM(**filter_none_values(kwargs))
        print(
            "DEBUGPRINT[89]: llm_service.py:109 (after llm: ChatLiteLLM = ChatLiteLLM(**filter_…)"
        )

        system_prompt: str = FOSRA_SYSTEM_PROMPT

        instruction_text = "Please provide a detailed, comprehensive answer to the user's question using the information from their personal knowledge sources. Make sure to cite all information appropriately and engage in a conversational manner."

        human_message_content: str = f"""
        {source_content}
        User's question:
        <user_query>
            {newest_message.content}
        </user_query>
        
        {instruction_text}
        """

        lc_messages.append(
            SystemMessage(content=system_prompt),
        )
        lc_messages.append(
            HumanMessage(content=human_message_content),
        )

        print(lc_messages)

        print(
            "DEBUGPRINT[90]: llm_service.py:135 (before return llm.astream(input=lc_messages))"
        )
        return llm.astream(input=lc_messages)

    @staticmethod
    async def generate_chunk_summaries(parent_chunks: list[Chunk]):

        # NOTE: Better to do all at once or one at a time?
        # TODO: Pull in existing filters from the DB

        # TODO: Pass 3 Parent Chunks at a time to LLM For Classification

        # TODO: Accumulate LLM Filter Classification in List

        # TODO: Pass all Generated Filters to LLM For Consolidation

        # TODO: Return 2-3 Filters

        pass

    @staticmethod
    async def generate_graph_query(parent_chunks: list[Chunk]):
        return

    @staticmethod
    async def generate_graph_search(parent_chunks: list[Chunk]):
        return

    @staticmethod
    async def split_to_subqueries(parent_chunks: list[Chunk]):
        return

    @staticmethod
    async def check_coverage(parent_chunks: list[Chunk]):
        return

    @staticmethod
    async def summarize_document():
        return

    @staticmethod
    async def _summarize_single(doc: str):
        MOCK_TOPICS = [
            # TanStack Query / general data fetching docs
            "data_fetching",
            "cache_invalidation",
            "cache_mutation",
            "configuration",
            "pagination",
            "error_handling",
            "optimistic_updates",
            "prefetching",
            "query_filters",
            "subscriptions",
            "devtools",
            "ssr_hydration",
            # Source code structural
            "authentication",
            "data_transformation",
            "file_io",
            "api_client",
            "middleware",
            "database_access",
            "validation",
            "event_handling",
            "state_management",
            "routing",
            # Meta / navigation — always present
            "navigation",
        ]

        # TODO: Return 2-3 Filters
        config: LLMConfig = LLMConfig(
            config_id=0,
            config_name="the config name",
            provider="openrouter",
            model="qwen/qwen3.5-27b",
            api_key="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",
            api_base="https://openrouter.ai/api/v1",
        )

        model_string: str = build_model_string(
            provider=config.provider,
            model_name=config.model,
            custom_provider=config.custom_provider,
        )

        kwargs: dict[str, Any] = {
            "model": model_string,
            "api_key": config.api_key,
            "streaming": True,
        }

        if config.api_base:
            kwargs["api_base"] = config.api_base

        if config.litellm_params:
            kwargs.update(config.litellm_params)

        llm: ChatLiteLLM = ChatLiteLLM(**filter_none_values(kwargs))

        prompt = DOC_TOPIC_GEN_PROMPT.format(
            existing_topics=MOCK_TOPICS, chunk_text=doc
        )
        return llm.astream(input=prompt)

    async def _summarize_with_context(parent_chunks: list[Chunk]):
        # TODO: Pull in existing filters from the DB
        MOCK_TOPICS = [
            # TanStack Query / general data fetching docs
            "data_fetching",
            "cache_invalidation",
            "cache_mutation",
            "configuration",
            "pagination",
            "error_handling",
            "optimistic_updates",
            "prefetching",
            "query_filters",
            "subscriptions",
            "devtools",
            "ssr_hydration",
            # Source code structural
            "authentication",
            "data_transformation",
            "file_io",
            "api_client",
            "middleware",
            "database_access",
            "validation",
            "event_handling",
            "state_management",
            "routing",
            # Meta / navigation — always present
            "navigation",
        ]

        # TODO: Pass 3 Parent Chunks at a time to LLM For Classification

        # TODO: Accumulate LLM Filter Classification in List

        # TODO: Pass all Generated Filters to LLM For Consolidation

        # TODO: Return 2-3 Filters

        pass

    async def _summarize_all(parent_chunks: list[Chunk]):
        # TODO: Pull in existing filters from the DB
        MOCK_TOPICS = [
            # TanStack Query / general data fetching docs
            "data_fetching",
            "cache_invalidation",
            "cache_mutation",
            "configuration",
            "pagination",
            "error_handling",
            "optimistic_updates",
            "prefetching",
            "query_filters",
            "subscriptions",
            "devtools",
            "ssr_hydration",
            # Source code structural
            "authentication",
            "data_transformation",
            "file_io",
            "api_client",
            "middleware",
            "database_access",
            "validation",
            "event_handling",
            "state_management",
            "routing",
            # Meta / navigation — always present
            "navigation",
        ]

        # TODO: Pass 3 Parent Chunks at a time to LLM For Classification

        # TODO: Accumulate LLM Filter Classification in List

        # TODO: Pass all Generated Filters to LLM For Consolidation

        # TODO: Return 2-3 Filters

        pass
