from backend.src.domain.schemas.config import (
    ChunkerConfig,
    EmbedderConfig,
    LLMConfig,
    ParserConfig,
    RerankerConfig,
    UserPreferences,
    VectorStoreConfig,
)

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
    chunker=ChunkerConfig(),
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
    chunker=ChunkerConfig(),
)
