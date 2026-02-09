import sys
import time
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from litellm import CustomStreamWrapper
from litellm.types.utils import ModelResponse

from backend.src.api.schemas.api_schemas import UIMessage
from backend.src.domain.schemas import (
    ChunkerConfig,
    EmbedderConfig,
    LLMConfig,
    ParserConfig,
    RerankerConfig,
    StorageConfig,
    UserPreferences,
    VectorStoreConfig,
)
from backend.src.services.conversation.llm_service import LLMService


async def generate_stream(query):
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

    prefs.llm_default = llm_conf

    from litellm import acompletion

    response: ModelResponse | CustomStreamWrapper = await acompletion(
        model="ollama/mistral-nemo:12b",
        messages=[{"content": f"{query}", "role": "user"}],
        api_base="http://localhost:11434",
        stream=True,
    )

    if isinstance(response, ModelResponse) or not response:
        raise ValueError()

    async for chunk in response:
        content = chunk["choices"][0]["delta"].get("content", "")

        if content:
            sys.stdout.write(content)
            yield content
