from typing import Annotated
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from loguru import logger
from pydantic import BaseModel
from backend.src.api.dependencies import get_db_session, get_infra 
from backend.src.api.request_context import RequestContext
from backend.src.api.schemas import SourceResponseDeep
from backend.src.domain.schemas import ParserConfig, ChunkerConfig, EmbedderConfig
from backend.src.domain.schemas.config_schemas import VectorStoreConfig
from backend.src.storage.utils.converters import domain_to_response, pydantic_to_domain

from backend.src.tasks.processing import chunk_sources
from backend.src.tasks.processing import embed_documents
from backend.src.tasks.processing import parse_files
from backend.src.tasks.storing import store_file_vectors


router = APIRouter(prefix="/testing", tags=["Testing"])


# @router.post("/test_parse/{user_id}")
# async def parse_test(
#     request: Request,
#     user_id: str,
#     workspace_id: Annotated[str, Body()],
#     path: Annotated[str, Body()],
#     session=Depends(get_db_session),
# ):
#     context = RequestContext.create_simple(user_id=user_id, workspace_id=workspace_id)
#
#     logger.success(f"Successfully parsed {path}")
#
#
# @router.get("/available-parsers}")
# async def available_parsers(
#     request: Request,
#     user_id: str,
#     session=Depends(get_db_session),
# ):
#     from backend.src.services.processing.parser_service import ParserService
#
#     context = RequestContext.create_simple(user_id=user_id, workspace_id="999999")
#
#     parsers = ParserService.get_available_parsers()
#
#     logger.success(f"Successfully got parsers {len(parsers)}")
#
#     return parsers


class UploadPathsRequest(BaseModel):
    paths: list[str]


# @router.post("/test_raw/")
# async def test_ingestions(
#     paths: UploadPathsRequest,
#     ctx: RequestContext = Depends(get_request_context),
#     infra=Depends(get_infra),
# ):
#     from backend.src.tasks.ingesting import batch_ingest_documents
#
#     logger.debug(ctx.preferences.storage)
#
#     try:
#         success = await batch_ingest_documents(
#             file_paths=paths.paths,
#             session_factory=infra.session_factory,
#             ctx=ctx,
#         )
#
#         parse = await parse_files(
#             files_list=success,
#             config=pydantic_to_domain(ctx.preferences.parser, ParserConfig),
#             session_factory=infra.session_factory,
#         )
#
#         # logger.info(f"FASTAPI ROUTE: Parsed {len(parse)} sources, starting chunking...")
#
#         chunked = await chunk_sources(
#             sources=parse,
#             config=pydantic_to_domain(ctx.preferences.chunker, ChunkerConfig),
#         )
#
#         # logger.info(f"FASTAPI ROUTE: Chunked {len(chunked)} sources.")
#         # logger.info("FASTAPI ROUTE: Converting chunked sources to response objects...")
#
#         source_objects = [domain_to_response(i, SourceResponseDeep) for i in chunked]
#
#         # logger.info("FASTAPI ROUTE: Conversion complete.")
#         # logger.info("FASTAPI ROUTE: Starting embedding of chunked sources...")
#
#         embedded_sources = await embed_documents(
#             sources=chunked,
#             config=pydantic_to_domain(ctx.preferences.embedder, EmbedderConfig),
#             session_factory=infra.session_factory,
#         )
#
#         # logger.info(f"FASTAPI ROUTE: Embedded {len(embedded_sources)} sources.")
#
#         api_embedded_sources = [
#             domain_to_response(i, SourceResponseDeep) for i in embedded_sources
#         ]
#
#         # logger.info("FASTAPI ROUTE: Storing embedded vectors...")
#
#         vec_config = pydantic_to_domain(ctx.preferences.vector_store, VectorStoreConfig)
#
#         store_vectors = await store_file_vectors(
#             sources=embedded_sources,
#             config=vec_config,
#             session_factory=infra.session_factory,
#         )
#
#         # logger.info(f"FASTAPI ROUTE: Stored vectors for {len(store_vectors)} sources.")
#         # logger.info("FASTAPI ROUTE: Ingestion test completed successfully.")
#
#         # WARN: MUST TRANSLATE TO RESPONSE OBJECTS ON OUTPUT
#         return store_vectors
#
#     except Exception as e:
#         logger.error(f"Unexpected error during ingestion test: {e}")
#         raise HTTPException(
#             status_code=500, detail=f"Internal Server Error: {e}"
#         ) from e
#
#
# # @router.post("/test_upload/{user_id}")
# # async def batch_file_upload(
# #     request: Request,
# #     paths: UploadPathsRequest,
# #     ctx: RequestContext = Depends(get_request_context),
# #     session=Depends(get_db_session),
# #     infra=Depends(get_infra),
# # ):
# #     from backend.src.tasks.ingesting import (
# #         batch_ingest_documents as task_batch_file_upload,
# #     )
# #
# #     success = await task_batch_file_upload(
# #         paths_list=paths.paths,
# #         session=session,
# #         ctx=ctx,
# #         infra=infra,
# #     )
# #
# #     return {"success": success}
