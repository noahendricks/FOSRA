from __future__ import annotations
from pymilvus.grpc_gen.common_pb2 import Retrieve
import datetime

import json
from typing import TYPE_CHECKING, Any

import litellm
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatLiteLLM
from loguru import logger
import pytest

from backend.src.api.schemas.api_schemas import (
    UIMessage,
    UIMessagePart,
    TextPart,
    FilePart,
)
from backend.src.domain.exceptions import (
    LLMConfigurationError,
    LLMValidationError,
)
from backend.src.domain.schemas import (
    ChunkerConfig,
    EmbedderConfig,
    LLMConfig,
    ParserConfig,
    RerankerConfig,
    RetrievedResult,
    SourceFull,
    StorageConfig,
    ValidationResult,
    VectorStoreConfig,
)
from backend.src.domain.enums import LLMRole
from backend.src.domain.schemas.config_schemas import UserPreferences
from backend.src.domain.schemas.source_schemas import ChunkWithScore, SourceGroup
from backend.src.services.conversation.prompts import FOSRA_CITATION_INSTRUCTIONS 

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

# =============================================================================
# Helper Functions
# =============================================================================


def filter_none_values(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def build_model_string(
    provider: str,
    model_name: str,
    custom_provider: str | None = None,
) -> str:
    if not provider or not model_name:
        raise LLMConfigurationError(
            reason="Provider and model name are required",
            remediation="Provide valid provider and model name",
        )

    if custom_provider:
        model_string = f"{custom_provider}/{model_name}"
        logger.debug(f"Built custom model string: {model_string}")
        return model_string

    provider_upper = provider.upper()

    if provider_upper == "OPENROUTER":
        model_string = f"openrouter/{model_name}"
        logger.debug(f"Built OpenRouter model string: {model_string}")
        return model_string

    prefix = PROVIDER_TO_LITELLM_MAP.get(provider_upper, provider.lower())
    model_string = f"{prefix}/{model_name}"

    logger.debug(f"Built model string: {model_string} for provider {provider}")
    return model_string


# =============================================================================
# LLM Service Class
# =============================================================================


class LLMService:
    @staticmethod
    async def validate_config(
        llm_config: LLMConfig,
        timeout: int = 30,
    ) -> ValidationResult:
        logger.info(
            f"Validating LLM configuration: {llm_config.provider}/{llm_config.model}"
        )

        try:
            if not llm_config.provider or not llm_config.model:
                raise LLMConfigurationError(
                    reason="Provider and model are required",
                    remediation="Provide complete LLM configuration",
                )

            if not llm_config.api_key:
                raise LLMConfigurationError(
                    reason="API key is required",
                    remediation="Provide valid API key in configuration",
                )

            model_string = build_model_string(
                provider=llm_config.provider,
                model_name=llm_config.model,
                custom_provider=llm_config.custom_provider,
            )

            kwargs: dict[str, Any] = {
                "model": model_string,
                "api_key": llm_config.api_key,
                "timeout": timeout,
            }

            if llm_config.api_base:
                kwargs["api_base"] = llm_config.api_base

            if llm_config.litellm_params:
                kwargs.update(llm_config.litellm_params)

            llm = ChatLiteLLM(**kwargs)  # pyright: ignore

            response = await llm.ainvoke(input="Hello", timeout=20)

            clean_content = str(response.content).strip()

            if not clean_content:
                logger.info(f"LLM returned empty response for {llm_config.model}")
                await logger.complete()
                return ValidationResult(is_valid=False, error_message="Empty response")

            logger.info(
                f"LLM validation successful for {llm_config.provider}/{llm_config.model}"
            )

            result = ValidationResult(
                is_valid=True,
                response_preview=str(response.content)[:100],
            )
            await logger.complete()

            return result
        except LLMConfigurationError:
            raise
        except Exception as e:
            logger.error(
                f"LLM validation failed for {llm_config.provider}/{llm_config.model}: {e}"
            )
            raise LLMValidationError(
                provider=llm_config.provider,
                model=llm_config.model,
                reason=str(e),
            ) from e

    @staticmethod
    def _create_llm_from_config(config: LLMConfig) -> ChatLiteLLM:
        """Create a ChatLiteLLM instance from configuration."""

        try:
            model_string = build_model_string(
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

            llm = ChatLiteLLM(**filter_none_values(kwargs))  # pyright: ignore
            logger.debug(f"Created LLM instance: {model_string}")

            return llm

        except Exception as e:
            logger.error(f"Failed to create LLM from config: {e}")
            raise LLMConfigurationError(
                reason=str(e),
                llm_role="unknown",
                remediation="Check LLM configuration parameters",
            ) from e

    @staticmethod
    def get_all_role_llms(
        config: UserPreferences,
    ) -> dict[LLMRole, ChatLiteLLM | None]:
        logger.info("Creating LLM instances for all roles")

        result: dict[LLMRole, ChatLiteLLM | None] = {
            LLMRole.HEAVY: None,
            LLMRole.FAST: None,
            LLMRole.LOGICAL: None,
        }

        try:
            if heavy_config := config.get_llm_config(LLMRole.HEAVY):
                result[LLMRole.HEAVY] = LLMService._create_llm_from_config(heavy_config)

            if fast_config := config.get_llm_config(LLMRole.FAST):
                result[LLMRole.FAST] = LLMService._create_llm_from_config(fast_config)

            if logical_config := config.get_llm_config(LLMRole.LOGICAL):
                result[LLMRole.LOGICAL] = LLMService._create_llm_from_config(
                    logical_config
                )

            return result

        except Exception as e:
            logger.error(f"Error getting role LLMs: {e}")
            return result

    @staticmethod
    def get_available_providers() -> list[str]:
        return list(PROVIDER_TO_LITELLM_MAP.keys())

    @staticmethod
    async def test_connection(
        provider: str,
        model: str,
        api_key: str,
        api_base: str | None = None,
        timeout: int = 10,
    ) -> dict[str, Any]:
        logger.info(f"Testing connection to {provider}/{model}")

        try:
            model_string = build_model_string(
                provider=provider,
                model_name=model,
            )

            kwargs: dict[str, Any] = {
                "model": model_string,
                "api_key": api_key,
                "timeout": timeout,
            }

            if api_base:
                kwargs["api_base"] = api_base

            llm: ChatLiteLLM = ChatLiteLLM(**filter_none_values(kwargs))  # pyright: ignore

            response = await llm.ainvoke([HumanMessage(content="test")])

            if response:
                logger.success(f"Connection test successful for {provider}/{model}")
                return {
                    "success": True,
                    "provider": provider,
                    "model": model,
                    "response_length": len(str(response.content)),
                }

            return {
                "success": False,
                "provider": provider,
                "model": model,
                "error": "Empty response",
            }

        except Exception as e:
            logger.error(f"Connection test failed for {provider}/{model}: {e}")
            return {
                "success": False,
                "provider": provider,
                "model": model,
                "error": str(e),
            }


llm_conf = LLMConfig(
    config_id=0,
    config_name="the config name",
    provider="ollama",
    model="mistral-nemo:12b",
    api_key="anrhsienrtashaetrn",
    api_base="http://localhost:11434",
    language="nerg",
)

llm_conf_heavy = LLMConfig(
    config_id=0,
    config_name="the config name",
    provider="ollama",
    model="mistral-nemo:12b",
    api_key="anrhsienrtashaetrn",
    api_base="http://localhost:11434",
    language="nerg",
)

llm_conf_fast = LLMConfig(
    config_id=0,
    config_name="the config name",
    provider="ollama",
    model="mistral-nemo:12b",
    api_key="anrhsienrtashaetrn",
    api_base="http://localhost:11434",
    language="nerg",
)

llm_conf_logic = LLMConfig(
    config_id=0,
    config_name="the config name",
    provider="ollama",
    model="mistral-nemo:12b",
    api_key="anrhsienrtashaetrn",
    api_base="http://localhost:11434",
    language="nerg",
)

prefs = UserPreferences(
    llm_default=llm_conf,
    llm_heavy=llm_conf_heavy,
    llm_fast=llm_conf_fast,
    llm_logic=llm_conf_logic,
    vector_store=VectorStoreConfig(),
    embedder=EmbedderConfig(),
    parser=ParserConfig(),
    reranker=RerankerConfig(),
    storage=StorageConfig(),
    chunker=ChunkerConfig(),
)


def _build_chat_history_section(chat_history: str | None = None):
    if chat_history:
        return f"""
!chat_history!
{chat_history if chat_history else "NO CHAT HISTORY PROVIDED"}
!chat_history!
"""
    return """
!chat_history!
NO CHAT HISTORY PROVIDED
!chat_history!
"""


def _extract_text_from_parts(parts: list[UIMessagePart]) -> str:
    text_parts: list[str] = []
    file_parts: dict[str, dict[str, Any]] = {}
    for part in parts:
        if isinstance(part, TextPart) and part.type == "text":
            text_parts.append(part.text if part.text else "")
        if isinstance(part, FilePart) and part.type == "file":
            file_parts[part.filename if part.filename else ""] = {
                "url": part.url,
                "mediaType": part.media_type,
            }

    return "\n".join(text_parts)


def ui_message_to_lc_message(ui_messages: list[UIMessage]) -> list[BaseMessage]:
    lc_messages: list[BaseMessage] = []
    for msg in ui_messages:
        if msg.role == "user":
            lc_messages.append(
                HumanMessage(content=_extract_text_from_parts(msg.parts))
            )
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=_extract_text_from_parts(msg.parts)))
    return lc_messages


def langchain_chat_history_to_str(chat_history: list[BaseMessage]) -> str:
    chat_history_str = ""
    for chat_message in chat_history:
        if isinstance(chat_message, HumanMessage):
            chat_history_str += f"<user>{chat_message.content}</user>\n"
        elif isinstance(chat_message, AIMessage):
            chat_history_str += f"<assistant>{chat_message.content}</assistant>\n"
        elif isinstance(chat_message, SystemMessage):
            chat_history_str += f"<system>{chat_message.content}</system>\n"
    return chat_history_str


def _format_system_prompt(
    prompt_template: str,
    chat_history: str | None = None,
) -> str:
    date: str = datetime.datetime.now().strftime(format="%Y-%m-%d")
    chat_history_section: str = _build_chat_history_section(chat_history)

    return prompt_template.format(
        date=date,
        chat_history_section=chat_history_section,
    )


llm_conf: LLMConfig = LLMConfig(
    config_id=0,
    config_name="default",
    provider="ollama",
    model="mistral-nemo:12b",
    api_key="sk-...",
    api_base="http://localhost:11434",
    language="en",
)
prefs: UserPreferences = UserPreferences(
    llm_default=llm_conf,
    llm_heavy=llm_conf,
    llm_fast=llm_conf,
    llm_logic=llm_conf,
    vector_store=VectorStoreConfig(),
    embedder=EmbedderConfig(),
    parser=ParserConfig(),
    reranker=RerankerConfig(),
    storage=StorageConfig(),
    chunker=ChunkerConfig(),
)


def format_source_for_citation(source: SourceGroup) -> str:
    def _to_cdata(value: Any) -> str:
        text: str = "" if value is None else str(object=value)
        return "<![CDATA[" + text.replace("]]>", "]]]]><![CDATA[>") + "]]>"

    # NOTE: Currently using whole source object as metadata; later dedicated metadata usage

    metadata: SourceFull = source.source

    source_id: str = metadata.source_id

    title: str = metadata.name

    sources_json: str = json.dumps(source.to_dict())

    document_type = (
        metadata.document_type
        if metadata.document_type
        else "No Document Type Provided"
    )

    url: str = metadata.origin_path

    metadata_json: str = json.dumps(obj=sources_json, ensure_ascii=False)

    chunks: list[ChunkWithScore] = source.chunks

    if not chunks:
        logger.debug(f"No Chunks for Source: {metadata.name}")

    chunks_xml: str = "\n".join(
        [
            f"<chunk id='{chunk_pkg.chunk.chunk_id}'>{_to_cdata(value=chunk_pkg.chunk.text)}</chunk>"
            for chunk_pkg in chunks
        ]
    )

    return f"""<document>
<document_metadata>
<document_id>{source_id}</document_id>
<document_type>{document_type}</document_type>
<title>{_to_cdata(title)}</title>
<url>{_to_cdata(url)}</url>
<metadata_json>{_to_cdata(metadata_json)}</metadata_json>
</document_metadata>

<document_content>
{chunks_xml}
</document_content>
</document>"""


def format_sources_section(
    sources: list[SourceGroup],
    section_title: str = "Source material",
) -> str:
    """Format multiple documents into a complete documents section."""
    if not sources:
        return ""

    formatted_sources: list[str] = [
        format_source_for_citation(source=src) for src in sources
    ]

    return f"""{section_title}:
    <documents>
    {chr(10).join(formatted_sources)}
    </documents>"""


async def generate_llm_response(
    chat_history: list[UIMessage],
    sources: list[SourceGroup],
    convo_id: str,
    user_prefs: UserPreferences | None,
):
    lc_messages: list[BaseMessage] = ui_message_to_lc_message(ui_messages=chat_history)
    chat_history_str: str = langchain_chat_history_to_str(chat_history=lc_messages)

    source_text: str = format_sources_section(sources)

    if not lc_messages:
        raise ValueError("No messages provided to generate response")

    newest_message: BaseMessage = lc_messages[-1]

    config: LLMConfig = LLMConfig(
        config_id=0,
        config_name="the config name",
        provider="ollama",
        model="mistral-nemo:12b",
        api_key="anrhsienrtashaetrn",
        api_base="http://localhost:11434",
        language="nerg",
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

    from pympler import asizeof

    asizeof.asizeof(chat_history)

    if config.api_base:
        kwargs["api_base"] = config.api_base

    if config.litellm_params:
        kwargs.update(config.litellm_params)

    llm: ChatLiteLLM = ChatLiteLLM(**filter_none_values(kwargs))  # pyright: ignore

    if not llm:
        error_message = f"No fast LLM configured for workspace {''}"
        logger.error(error_message)
        raise RuntimeError(error_message)

    system_prompt: str = _format_system_prompt(
        prompt_template=DEFAULT_QNA_NO_DOCUMENTS_PROMPT,
        chat_history=chat_history_str,
    )

    instruction_text = "Please provide a detailed, comprehensive answer to the user's question using the information from their personal knowledge sources. Make sure to cite all information appropriately and engage in a conversational manner."

    human_message_content: str = f"""
    {source_text}
    User's question:
    <user_query>
        {newest_message.content}
    </user_query>
    
    {instruction_text}
    """

    messages_with_chat_history: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message_content),
    ]

    return llm.astream(input=messages_with_chat_history)
