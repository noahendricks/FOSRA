from __future__ import annotations

import asyncio
from loguru import logger

from FOSRABack.src.api.dependencies import get_db_session
from FOSRABack.src.storage.repos.config_repo import ConfigRepository
from FOSRABack.src.domain.schemas.schemas import ToolConfig
from FOSRABack.src.domain.enums import StorageBackendType
from FOSRABack.src.storage.models import ToolCategory


# =============================================================================
# System Default Configurations
# =============================================================================

SYSTEM_DEFAULTS = [
    ToolConfig(
        name="Default Primary LLM",
        description="System default for primary LLM operations",
        category=ToolCategory.LLM,
        provider="openrouter",
        model="openai/gpt-4o-mini",
        details={
            "api_key": "YOUR_API_KEY_HERE",
            "temperature": 0.7,
            "max_tokens": 4000,
        },
        is_system_default=True,
    ),
    ToolConfig(
        name="Default Fast LLM",
        description="System default for fast LLM operations",
        category=ToolCategory.LLM,
        provider="openrouter",
        model="openai/gpt-3.5-turbo",
        details={
            "api_key": "YOUR_API_KEY_HERE",
            "temperature": 0.5,
            "max_tokens": 2000,
        },
        is_system_default=True,
    ),
    ToolConfig(
        name="Default Vector Store",
        description="System default Qdrant instance",
        category=ToolCategory.VECTOR_STORE,
        provider="qdrant",
        model=None,
        details={
            "host": "localhost",
            "port": 6333,
            "collection_name": "default",
            "top_k": 10,
        },
        is_system_default=True,
    ),
    ToolConfig(
        name="Default Embedder",
        description="System default embedding model",
        category=ToolCategory.EMBEDDER,
        provider="fastembed",
        model="BAAI/bge-small-en-v1.5",
        details={
            "batch_size": 32,
            "normalize": True,
        },
        is_system_default=True,
    ),
    ToolConfig(
        name="Default Parser",
        description="System default document parser",
        category=ToolCategory.PARSER,
        provider="docling",
        model=None,
        details={
            "extract_tables": True,
            "extract_images": False,
            "ocr_enabled": True,
        },
        is_system_default=True,
    ),
    ToolConfig(
        name="Default Reranker",
        description="System default reranker",
        category=ToolCategory.RERANKER,
        provider="cohere",
        model="rerank-english-v2.0",
        details={
            "top_k": 5,
            "enabled": False,
        },
        is_system_default=True,
    ),
    ToolConfig(
        name="Default Storage Backend",
        description="System default file storage",
        category=ToolCategory.STORAGE,
        provider="filesystem",
        model=None,
        details={
            "backend_type": StorageBackendType.FILESYSTEM,
            "base_path": "/data/uploads",
            "timeout_seconds": 30,
            "max_retries": 3,
            "chunk_size": 8192,
        },
        is_system_default=True,
    ),
]


async def set_system_defaults():
    """Create all system default configurations."""
    # FIX:
    async with get_db_session() as session:
        logger.info("Setting up system default configurations...")

        for config_data in SYSTEM_DEFAULTS:
            try:
                existing = await ConfigRepository.get_system_default(
                    session, config_data.category
                )

                if existing:
                    logger.warning(
                        f"System default for {config_data.category} already exists, skipping"
                    )
                    continue

                config = await ConfigRepository.create_tool_config(
                    session=session,
                    user_id="SYSTEM",
                    config_data=config_data,
                )

                logger.success(
                    f"Created system default: {config.name} ({config.category})"
                )

            except Exception as e:
                logger.error(
                    f"Failed to create system default for {config_data.category}: {e}"
                )

        logger.info("System defaults setup complete!")


if __name__ == "__main__":
    asyncio.run(set_system_defaults())
