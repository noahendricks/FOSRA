from loguru import logger
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from FOSRABack.src.settings import settings
from FOSRABack.src.storage.models import Base


class Infrastructure:
    """Holds heavy singletons. Initialized ONCE at startup."""

    def __init__(self, settings):
        self.qdrant_client: AsyncQdrantClient | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None
        self.engine = create_async_engine(settings.database.url, echo=True)
        self._tables_created = False

    def init(self):
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

        self.qdrant_client = AsyncQdrantClient(
            location=":memory:",
        )

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

    async def close(self):
        if self.qdrant_client:
            await self.qdrant_client.close()

        if self.engine:
            await self.engine.dispose()
            logger.info("Database engine disposed.")

        logger.info("Infrastructure cleanup complete.")


global_infra = Infrastructure(settings)
logger.info(f"Settings loaded: {settings.dict()}")
