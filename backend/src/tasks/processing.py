from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .broker import broker

# NOTE: The following tasks use phantom types (SourceFull, FileContent, ParserConfig, ParserService)
# that do not exist in the codebase. Commenting out until properly implemented.

# @broker.task
# async def parse_files(
#     files_list: list[FileContent],
#     config: ParserConfig,
#     session_factory: async_sessionmaker[AsyncSession],
# ) -> list[SourceFull]:
#     """
#     process FileContent to source object
#
#     Returns:
#         FileContent object with raw file bytes and metadata
#     """
#     try:
#
#         parsed_files = await ParserService.parse_documents(
#             files=files_list, config=config, session_factory=session_factory
#         )
#
#         return parsed_files
#
#     except Exception as e:
#         logger.error(f"Parsing Task failed: {e}")
#         raise
#
#     finally:
#         pass


# @broker.task
# async def chunk_sources(
#     sources: list[SourceFull], config: ChunkerConfig
# ) -> list[SourceFull]:
#     """Chunk a list of sources in place"""
#
#     try:
#         logger.info(f"Starting Chunking Task for {len(sources)} sources")
#         chunked_sources = await ChunkerService().chunk_sources(
#             sources=sources,
#             config=config,
#         )
#         logger.info(f"Completed Chunking Task for {len(sources)} sources")
#
#         return chunked_sources
#
#     except Exception as e:
#         logger.error(f"Chunking Task failed: {e}")
#         raise


# @broker.task
# async def embed_documents(
#     sources: list[SourceFull],
#     config: EmbedderConfig,
#     session_factory: async_sessionmaker[AsyncSession],
# ) -> list[SourceFull]:
#     """
#     Process a file_path to raw bytes and relevant metadata"""
#     try:
#         logger.info(f"Starting embedding for {len(sources)} sources")
#
#         embedded = await EmbedderService().embed_sources(
#             sources=sources, config=config, session_factory=session_factory
#         )
#
#         # print(embedded[0])
#         logger.info(f"Completed embedding for {len(embedded)} sources")
#
#         return embedded
#
#     except Exception as e:
#         logger.error(f"Embedding Documents task failed: {e}")
#         raise


# @broker.task
# async def embed_query(
#     query: str,
#     config,
#     session: AsyncSession,
# ) -> list[float]:
#     """
#     Process a file_path to raw bytes and relevant metadata"""
#     from backend.src.services.processing.embedder_service import EmbedderService
#
#     try:
#         embedded = await EmbedderService().embed_query(query=query, config=config)
#         return embedded
#
#     except Exception as e:
#         logger.error(f"Embedding Query task failed :{e}")
#         raise
