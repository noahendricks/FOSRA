import json
from langchain_litellm import ChatLiteLLM
import litellm
import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from backend.src.domain.enums import LLMRole
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
from backend.src.services.conversation.llm_service import (
    LLMService,
    build_model_string,
)

from litellm.types.utils import ModelResponse


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def llm_service():
    return LLMService()


class TestLiteLLMInputs:
    def test_build_model_string(self):
        llm_conf = LLMConfig(
            config_id=0,
            config_name="the config name",
            provider="openrouter",
            model="opus-4.5",
            api_key="anrhsienrtashaetrn",
            language="nerg",
            litellm_params={},
        )

        result = build_model_string(
            provider=llm_conf.provider,
            model_name=llm_conf.model,
        )

        assert result == "openrouter/opus-4.5"

    @pytest.mark.asyncio
    async def test_validate_config(self):
        llm_conf = LLMConfig(
            config_id=0,
            config_name="the config name",
            provider="ollama",
            model="mistral-nemo:12b",
            api_key="anrhsienrtashaetrn",
            api_base="http://localhost:11434",
            language="nerg",
        )

        validated = await LLMService().validate_config(llm_config=llm_conf)

        assert validated.is_valid
        assert validated.error_message == ""

    @pytest.mark.asyncio
    async def test_get_llm_by_role(self):
        llm_conf = LLMConfig(
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
            vector_store=VectorStoreConfig(),
            embedder=EmbedderConfig(),
            parser=ParserConfig(),
            reranker=RerankerConfig(),
            storage=StorageConfig(),
            chunker=ChunkerConfig(),
        )

        role = LLMRole.HEAVY

        prefs.llm_default = llm_conf

        is_role = LLMService().get_llm_for_role(role=role, user_prefs=prefs)

        assert isinstance(is_role, ChatLiteLLM)
        assert is_role.model == "ollama/mistral-nemo:12b"
        assert is_role.api_base == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_get_all_roles_llm(self):
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

        all_roles = LLMService().get_all_role_llms(config=prefs)

        assert all_roles

        assert isinstance(all_roles[LLMRole.HEAVY], ChatLiteLLM)
        assert isinstance(all_roles[LLMRole.FAST], ChatLiteLLM)
        assert isinstance(all_roles[LLMRole.LOGICAL], ChatLiteLLM)

        assert all_roles[LLMRole.HEAVY].model == "ollama/mistral-nemo:12b"
        assert all_roles[LLMRole.FAST].model == "ollama/mistral-nemo:12b"
        assert all_roles[LLMRole.LOGICAL].model == "ollama/mistral-nemo:12b"

    @pytest.mark.asyncio
    async def test_generate_stream(self):
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

        from litellm import acompletion, CustomStreamWrapper

        response: ModelResponse | CustomStreamWrapper = await acompletion(
            model="ollama/mistral-nemo:12b",
            messages=[{"content": "what is the gregorian calendar", "role": "user"}],
            api_base="http://localhost:11434",
            stream=True,
        )

        if isinstance(response, ModelResponse) or not response:
            raise ValueError()

        assert isinstance(response, CustomStreamWrapper)

        chunks_received = 0
        content_received = False

        async for chunk in response:
            chunks_received = chunks_received + 1
            print(f"DEBUG: {chunk}")

            choice = chunk["choices"][0]

            print(f"DEBUG:... choice {choice}")

            delta = choice.delta
            print(f"DEBUG:... delta {choice.delta}")

            print(f"DEBUG:... delta {delta}")

            content = delta.content
            print(f"DEBUG:... content {content}")

            if content:
                content_received = True
                assert isinstance(content, str)

            print(f"DEBUG:... chunks_received {chunks_received}")
            print(f"DEBUG:... content_received {chunks_received}")

        print(f"DEBUG:... outside content_received {content_received}")
        print(f"DEBUG:... outside chunks_received {chunks_received}")
        assert chunks_received > 0, "Stream empty"
        assert content_received, "No text content received"
