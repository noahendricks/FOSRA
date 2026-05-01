from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING, Any

import litellm
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from loguru import logger

from backend.src.api.schemas.tui_control_schemas import (
    FilePart,
    TextPart,
    UIMessage,
    UIMessagePart,
)
from backend.src.settings import LLMConfig, ScoredRetrieval

if TYPE_CHECKING:
    pass

litellm.drop_params = False  # Must be False for minimax to include api_key


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
    "OLLAMA": "ollama_chat",
    "MINIMAX": "minimax",
    "MINIMAX-CODING-PLAN": "minimax",
}

# =============================================================================
# Helper Functions
# =============================================================================


def _filter_none_values(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _build_model_string(
    provider: str,
    model_name: str,
    custom_provider: str | None = None,
) -> str:
    if not provider or not model_name:
        raise ValueError("No Provider or Model Name Provided")

    if custom_provider:
        model_string = f"{custom_provider}/{model_name}"
        logger.debug("Built custom model string: {}", model_string)
        return model_string

    # Special case: MiniMax uses openai-compatible API with minimax/ prefix
    provider_upper = provider.upper()
    if provider_upper in ("MINIMAX-CODING-PLAN", "MINIMAX"):
        # litellm expects minimax/ prefix
        model_string = f"minimax/{model_name}"
        logger.debug("Built MiniMax model string: {}", model_string)
        return model_string

    if provider_upper == "OPENROUTER":
        model_string = f"openrouter/{model_name}"
        logger.debug("Built OpenRouter model string: {}", model_string)
        return model_string

    prefix = PROVIDER_TO_LITELLM_MAP.get(provider_upper, provider.lower())
    model_string = f"{prefix}/{model_name}"
    logger.debug("Built model string: {} for provider {}", model_string, provider)
    return model_string


# =============================================================================
# LLM Service Class
# =============================================================================


async def _validate_config(
    llm_config: LLMConfig,
    timeout: int = 30,
):
    logger.info(
        "Validating LLM configuration: {}/{}", llm_config.provider, llm_config.model
    )

    try:
        if not llm_config.provider or not llm_config.model:
            raise ValueError("")

        if not llm_config.api_key:
            raise ValueError("No LLM Config API Key provided")

        model_string = _build_model_string(
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
            logger.info("LLM returned empty response for {}", llm_config.model)
            await logger.complete()
            return False

        logger.info(
            "LLM validation successful for {}/{}", llm_config.provider, llm_config.model
        )

        result = True

        await logger.complete()

        return result
    except Exception as e:
        logger.opt(exception=True).error(
            "LLM validation failed for {}/{}", llm_config.provider, llm_config.model
        )
        raise ValueError(e) from e


def build_llm(config: LLMConfig) -> ChatLiteLLM:
    """Create a ChatLiteLLM instance from configuration."""

    try:
        model_string = _build_model_string(
            provider=config.provider,
            model_name=config.model,
            custom_provider=config.custom_provider,
        )
        
        

        api_key_val = config.get_api_key_value()
        logger.debug("API key prefix: {}", api_key_val[:10] if api_key_val else "None")
        kwargs: dict[str, Any] = {
            "model": model_string,
            "api_key": api_key_val,
            "streaming": True,
        }

        if config.api_base:
            kwargs["api_base"] = config.api_base

        if config.litellm_params:
            kwargs.update(config.litellm_params)

        llm = ChatLiteLLM(**_filter_none_values(kwargs))  # pyright: ignore

        logger.debug("Created LLM instance: {} with api_base: {}", model_string, kwargs.get("api_base"))

        return llm

    except Exception as e:
        logger.opt(exception=True).error("Failed to create LLM from config")
        raise ValueError(e)


def get_available_providers() -> list[str]:
    return list(PROVIDER_TO_LITELLM_MAP.keys())


async def test_connection(
    provider: str,
    model: str,
    api_key: str,
    api_base: str | None = None,
    timeout: int = 10,
) -> dict[str, Any]:
    logger.info("Testing connection to {}/{}", provider, model)

    try:
        model_string = _build_model_string(
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

        llm: ChatLiteLLM = ChatLiteLLM(**_filter_none_values(kwargs))  # pyright: ignore

        response = await llm.ainvoke([HumanMessage(content="test")])

        if response:
            logger.success("Connection test successful for {}/{}", provider, model)
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
        logger.opt(exception=True).error(
            "Connection test failed for {}/{}", provider, model
        )
        return {
            "success": False,
            "provider": provider,
            "model": model,
            "error": str(e),
        }


def build_chat_history_section(chat_history: str | None = None):
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


def extract_text_from_parts(parts: list[UIMessagePart]) -> str:
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


def ui_messages_to_lc_messages(ui_messages: list[UIMessage]) -> list[BaseMessage]:
    lc_messages: list[BaseMessage] = []
    for msg in ui_messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=extract_text_from_parts(msg.parts)))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=extract_text_from_parts(msg.parts)))
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


def format_system_prompt(
    prompt_template: str,
    chat_history: str | None = None,
) -> str:
    date: str = datetime.datetime.now().strftime(format="%Y-%m-%d")
    chat_history_section: str = build_chat_history_section(chat_history)

    return prompt_template.format(
        date=date,
        chat_history_section=chat_history_section,
    )


def format_source_for_citation(source: ScoredRetrieval) -> str:
    def to_citation(value: Any) -> str:
        text: str = "" if value is None else str(object=value)
        return "<![CDATA[" + text.replace("]]>", "]]]]><![CDATA[>") + "]]>"

    source_id: str = source.doc_id

    sources_json: str = source.model_dump_json()

    # get document type

    metadata_json: str = json.dumps(obj=sources_json, ensure_ascii=False)

    source_xml: str = "\n".join(
        [f"<source_id='{source.doc_id}'>{to_citation(value=source.text)}</chunk>"]
    )

    return f"""<document>
    <document_metadata>
    <document_id>{source.doc_id}</document_id>
    <title>{to_citation(source.doc_title)}</title>
    <metadata_json>{to_citation(metadata_json)}</metadata_json>
    </document_metadata>

    <document_content>
    {source_xml}
    </document_content>
    </document>"""


def format_sources_section(
    sources: list[ScoredRetrieval],
    section_title: str = "Source material",
) -> str:
    """Format multiple documents into a complete documents section."""
    if not sources:
        return ""

    formatted_sources: list[str] = [
        format_source_for_citation(source=s) for s in sources
    ]

    return f"""{section_title}:
    <documents>
    {chr(10).join(formatted_sources)}
    </documents>"""
