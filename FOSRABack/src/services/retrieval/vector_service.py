from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from FOSRABack.src.domain.exceptions import (
    VectorRetrievalError,
    VectorStorageError,
)
from FOSRABack.src.domain.schemas import (
    RetrievedResult,
    SourceFull,
    VectorSearchResult,
    VectorStoreConfig,
)
from FOSRABack.src.services.retrieval.impls.vector_stores import BaseVectorStore

if TYPE_CHECKING:
    pass


class VectorService:
    @staticmethod
    async def initialize_store(
        store_config: VectorStoreConfig,
        session: AsyncSession,
    ) -> BaseVectorStore:
        logger.debug(f"Initializing vector store: {store_config.store_type}")

        try:
            store = BaseVectorStore.get_store(
                store_type=store_config.store_type,
                session=session,
            )

            if not store:
                raise VectorStorageError(
                    operation="initialize_store",
                    reason="Store type not found",
                    collection_name=store_config.collection_name,
                    remediation="Verify store type is registered and configuration is correct",
                )

            await store.initialize(config=store_config)

            logger.info(
                f"Successfully initialized {store_config.store_type} vector store"
            )

            return store

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise VectorStorageError(
                operation="initialize_store",
                reason="Initialization error",
                collection_name=store_config.collection_name,
                remediation="Check vector store connectivity and configuration",
            )

    @staticmethod
    async def search(
        query_vector: list[float],
        query_text: str,
        config: VectorStoreConfig,
        session: AsyncSession,
    ) -> list[RetrievedResult]:
        if not query_vector:
            raise VectorRetrievalError(
                query_text=query_text,
                query_vector=query_vector,
                reason="Empty query vector",
                remediation="Provide a valid non-empty query vector",
            )

        logger.debug(
            f"Performing vector search with {len(query_vector)} dimensions, "
            f"top_k={config.top_k}, min_score={config.min_score}"
        )

        try:
            store: BaseVectorStore = await VectorService().initialize_store(
                store_config=config,
                session=session,
            )

            results: list[RetrievedResult] = await store.search(
                query_vector=query_vector,
                config=config,
            )
            print(f"DEBUGPRINT[66]: vector_service.py:124: results={results}")

            # Add Query Text to each result for context
            for result in results:
                print(
                    "DEBUGPRINT[65]: vector_service.py:131 (before result.query_text = query_text)"
                )
                result.query_text = query_text
                print(f"DEBUGPRINT[62]: vector_service.py:131: result={result}")

            logger.success(
                f"Vector search completed: {len(results)} results from {config.collection_name}"
            )

            return results

        except VectorStorageError:
            raise

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise VectorRetrievalError(
                query_vector=query_vector,
                query_text=query_text,
                reason=f"Search error: {e}",
                remediation="Check vector store connectivity and query format",
            ) from e

        finally:
            if store:  # pyright: ignore
                await store.close()

    @staticmethod
    async def upsert(
        sources: list[SourceFull],
        config: VectorStoreConfig,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> list[str]:
        if not sources:
            logger.warning("No sources provided for upsert")
            return []

        total_chunks = sum(len(s.chunks) for s in sources if s.chunks)

        logger.debug(
            f"Upserting {len(sources)} sources with {total_chunks} total chunks "
            f"to {config.collection_name}"
        )

        store = None
        async with session_factory() as session:
            try:
                store = await VectorService.initialize_store(
                    store_config=config,
                    session=session,
                )

                logger.info(
                    f"Upserting {len(sources)} sources with {total_chunks} total chunks "
                )

                logger.info(f" Config: {config.to_dict()})")

                point_ids = await store.upsert(
                    sources=sources,
                    collection_name=config.collection_name,
                    config=config,
                )

                logger.info(
                    f"Successfully upserted {len(point_ids)} vectors to {config.collection_name}"
                )

                return point_ids

            except VectorStorageError:
                # Re-raise initialization errors
                raise
            except Exception as e:
                logger.error(f"Vector upsert failed: {e}")
                raise VectorStorageError(
                    operation="upsert",
                    reason=f"Upsert error: {e}",
                    remediation="Check vector store connectivity and source data",
                )
            finally:
                if store:
                    await store.close()

    @staticmethod
    async def delete(
        point_ids: list[str],
        config: VectorStoreConfig,
        session: AsyncSession,
    ) -> int:
        if not point_ids:
            logger.warning("No point IDs provided for deletion")
            return 0

        logger.debug(f"Deleting {len(point_ids)} vectors from {config.collection_name}")

        store = None
        try:
            store = await VectorService.initialize_store(
                store_config=config,
                session=session,
            )

            deleted_count = await store.delete(
                collection_name=config.collection_name,
                point_ids=point_ids,
            )

            logger.info(
                f"Successfully deleted {deleted_count} vectors from {config.collection_name}"
            )
            return deleted_count

        except VectorStorageError:
            # Re-raise initialization errors
            raise
        except Exception as e:
            logger.error(f"Vector deletion failed: {e}")
            raise VectorStorageError(
                operation="delete",
                reason=f"Delete error: {e}",
                remediation="Check vector store connectivity and point IDs",
            ) from e
        finally:
            if store:
                await store.close()

    @staticmethod
    async def get_collection_info(
        collection_name: str,
        config: VectorStoreConfig,
        session: AsyncSession,
    ) -> dict[str, Any]:
        logger.debug(f"Retrieving info for collection: {collection_name}")

        store = None
        try:
            store = await VectorService.initialize_store(
                store_config=config,
                session=session,
            )

            info = await store.get_collection_info(collection_name=collection_name)

            logger.info(f"Retrieved info for collection {collection_name}")
            return info

        except VectorStorageError:
            # Re-raise initialization errors
            raise
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise VectorRetrievalError(
                query_text="",
                query_vector=[],
                reason=f"Info retrieval error: {e}",
                remediation="Check vector store connectivity and collection name",
            ) from e
        finally:
            if store:
                await store.close()

    @staticmethod
    async def ensure_collection(
        collection_name: str,
        config: VectorStoreConfig,
        session: AsyncSession,
    ) -> None:
        logger.debug(
            f"Ensuring collection exists: {collection_name} "
            f"(vector_size={config.vector_size}, distance={config.distance_metric})"
        )

        store = None
        try:
            store = await VectorService.initialize_store(
                store_config=config,
                session=session,
            )

            await store.ensure_collection(
                collection_name=collection_name,
                config=config,
            )

            logger.info(f"Collection {collection_name} is ready")

        except VectorStorageError:
            # Re-raise initialization errors
            raise
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise VectorStorageError(
                operation="ensure_collection",
                reason=f"Collection creation error: {e}",
                collection_name=collection_name,
                remediation="Check vector store connectivity and configuration",
            ) from e
        finally:
            if store:
                await store.close()

    @staticmethod
    def get_available_stores() -> list[str]:
        return BaseVectorStore.get_available_stores()
