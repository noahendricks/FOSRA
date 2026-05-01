from typing import Any

from falkordb import FalkorDB
from loguru import logger
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.src.settings import settings
from backend.src.storage.models import Base


class Infrastructure:
    """holds heavy singletons. initialized ONCE at startup."""

    def __init__(self, settings: Any) -> None:  # type: ignore[reportExplicitAny]
        self.qdrant_client: AsyncQdrantClient | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None
        self.engine = create_async_engine(settings.database.url, echo=False)  # type: ignore[reportUnknownMemberType]
        self.falkordb_client: FalkorDB | None = None
        self.falkordb_graph: Any = None  # type: ignore[reportExplicitAny]
        self.model_registry: Any = None  # type: ignore[reportExplicitAny]
        self._tables_created = False
        self.checkpointer: Any = None

    def init(self):
        from backend.src.settings.fosra_paths import fosra_paths

        _ = fosra_paths.fosra

        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

        from backend.src.services.model_registry import ModelRegistry

        self.model_registry = ModelRegistry.get_instance()
        logger.info("ModelRegistry initialized")

        qdrant_settings = settings.qdrant

        if qdrant_settings.data_path:
            self.qdrant_client = AsyncQdrantClient(
                path=qdrant_settings.data_path,
            )
            logger.info(
                "Qdrant initialized with persistent storage at: {}",
                qdrant_settings.data_path,
            )
        elif qdrant_settings.url:
            self.qdrant_client = AsyncQdrantClient(
                url=qdrant_settings.url,
            )
            logger.info("Qdrant initialized with remote URL: {}", qdrant_settings.url)
        elif qdrant_settings.host:
            self.qdrant_client = AsyncQdrantClient(
                host=qdrant_settings.host,
                port=qdrant_settings.port,
            )
            logger.info(
                "Qdrant initialized at {}:{}",
                qdrant_settings.host,
                qdrant_settings.port,
            )
        else:
            self.qdrant_client = AsyncQdrantClient(location=":memory:")
            logger.warning(
                "Qdrant using in-memory mode. Set QDRANT__DATA_PATH for persistence."
            )

        falkordb_settings = settings.falkordb
        try:
            db = FalkorDB(
                host=falkordb_settings.host,
                port=falkordb_settings.port,
            )
            self.falkordb_client = db
            self.falkordb_graph = db.select_graph(falkordb_settings.graph_name)
            logger.info(
                "FalkorDB initialized at {}:{}/{}",
                falkordb_settings.host,
                falkordb_settings.port,
                falkordb_settings.graph_name,
            )
        except Exception as e:
            logger.warning("FalkorDB not available: {}. Graph features disabled.", e)
            self.falkordb_client = None
            self.falkordb_graph = None

        logger.info("Infrastructure initialized.")

    async def init_models(self):
        if self._tables_created:
            logger.debug("Database tables already created. Skipping...")
            return

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._tables_created = True

            logger.info("Database tables created successfully.")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise

        await self._init_session_state_manager()
        await self._init_checkpointer()
        await self._init_models_dev()

    async def _init_models_dev(self) -> None:
        """Initialize models.dev data fetching and caching."""
        try:
            from backend.src.api.schemas.provider_registry import _init_models_dev

            await _init_models_dev()
            logger.info("models.dev initialized")
        except Exception as e:
            logger.warning(
                "models.dev initialization failed: {}. Continuing without.", e
            )

    async def _init_session_state_manager(self):
        try:
            from backend.src.services.session.session_state_manager import (
                SessionStateManager,
            )

            manager = await SessionStateManager.get_instance()
            manager.set_session_factory(self.session_factory)

            logger.info("SessionStateManager initialized with database persistence.")
        except Exception as e:
            logger.warning(
                "SessionStateManager initialization failed: {}. Session persistence will be in-memory only.",
                e,
            )

    async def _init_checkpointer(self):
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            conn_string = "host=localhost port=5432 dbname=postgres user=postgres"
            async with AsyncPostgresSaver.from_conn_string(conn_string) as saver:
                self.checkpointer = saver
            logger.info("LangGraph Postgres checkpointer initialized.")
        except Exception as e:
            logger.warning(
                "Checkpointer initialization failed: {}. Agent will be stateless per turn.",
                e,
            )
            self.checkpointer = None

    async def close(self):
        if self.qdrant_client:
            await self.qdrant_client.close()

        if self.checkpointer is not None:
            try:
                await self.checkpointer.conn.close()
                logger.info("Checkpointer connection closed.")
            except Exception as e:
                logger.warning("Error closing checkpointer connection: {}", e)

        if self.engine:
            await self.engine.dispose()
            logger.info("Database engine disposed.")

        if self.falkordb_client:
            try:
                self.falkordb_client.close()
            except Exception as e:
                logger.warning("Error closing FalkorDB client: {}", e)

        logger.info("Infrastructure cleanup complete.")

    # Alias for close()
    async def cleanup(self):
        return await self.close()


global_infra = Infrastructure(settings)
