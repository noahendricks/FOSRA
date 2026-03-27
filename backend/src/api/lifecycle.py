import asyncio
import os

from loguru import logger
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from falkordb import FalkorDB
from backend.src.settings import settings
from backend.src.storage.models import Base


class Infrastructure:
    """Holds heavy singletons. Initialized ONCE at startup."""

    def __init__(self, settings):
        self.qdrant_client: AsyncQdrantClient | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None
        self.engine = create_async_engine(settings.database.url, echo=False)
        self.falkordb_client: FalkorDB | None = None
        self.falkordb_graph = None
        self.model_registry = None
        self._tables_created = False
        self.checkpointer = None

    def init(self):
        from backend.src.settings.fosra_dirs import fosra_dirs

        _ = fosra_dirs.fosra

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

    async def _init_session_state_manager(self):
        try:
            from backend.src.storage.repos.session_state_repo import SessionStateRepo
            from backend.src.services.session.session_state_manager import (
                SessionStateManager,
            )

            manager = await SessionStateManager.get_instance()
            async with self.session_factory() as session:
                repo = SessionStateRepo(session)
                manager.set_repo(repo)
            logger.info("SessionStateManager initialized with database persistence.")
        except Exception as e:
            logger.warning(
                "SessionStateManager initialization failed: {}. "
                "Session persistence will be in-memory only.",
                e,
            )

    async def _init_checkpointer(self):
        try:
            from langgraph.checkpoint.postgres import AsyncPostgresSaver

            conn_string = settings.database.url
            self.checkpointer = AsyncPostgresSaver.from_conn_string(conn_string)
            await self.checkpointer.setup()
            logger.info("LangGraph Postgres checkpointer initialized.")
        except Exception as e:
            logger.warning(
                "Checkpointer initialization failed: {}. "
                "Agent will be stateless per turn.",
                e,
            )
        self.checkpointer: AsyncPostgresSaver | None = None

    async def close(self):
        if self.qdrant_client:
            await self.qdrant_client.close()

        if self.engine:
            await self.engine.dispose()
            logger.info("Database engine disposed.")

        if self.falkordb_client:
            try:
                self.falkordb_client.close()
            except Exception as e:
                logger.warning("Error closing FalkorDB client: {}", e)

        logger.info("Infrastructure cleanup complete.")


global_infra = Infrastructure(settings)
