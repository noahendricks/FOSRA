# from datetime import datetime
# from pathlib import Path
# import pytest
# import pytest_asyncio
#
# from sqlalchemy import select
# from sqlalchemy.ext.asyncio import AsyncSession
# from FOSRABack.src.storage.database import SQLSessionManager
# from FOSRABack.src.storage.models import ChunkORM, SourceORM
# from unittest.mock import AsyncMock, MagicMock, patch
#
# from FOSRABack.src.processing.parse import process_source_in_background
# from FOSRABack.src.processing.utils.processing_utils import generate_content_hash
# from FOSRABack.src.storage.schemas import (
#     Chunk,
#     Source,
#     SourceWithRelations,
#     SparseVector,
# )
# from FOSRABack.src.storage.vector_service import VectorRepo
#
#
# @pytest_asyncio.fixture(scope="module")
# async def test_db_manager():
#     manager = SQLSessionManager("sqlite+aiosqlite:///:memory:")
#
#     await manager.init()
#
#     # Create all tables in the database
#     from FOSRABack.src.storage.models import Base
#
#     async with manager.engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
#
#     yield manager
#
#     await manager.dispose()
#
#
# @pytest_asyncio.fixture
# async def real_session(test_db_manager: SQLSessionManager):
#     async with test_db_manager.session_scope() as session:
#         yield session
#
#
# @pytest_asyncio.fixture
# async def mock_source():
#     m_source = Source(
#         unique_id="1234",
#         content_summary="this is content",
#         summary_embedding="0.1" * 384,
#         created_date=datetime.now(),
#         source_id="4321",
#         workspace_ids=[7],
#         uploaded_at=datetime.now(),
#         source_hash=None,
#     )
#     return m_source
#
#
# @pytest.mark.asyncio
# @patch("FOSRABack.src.processing.parse.state.vector_service")
# @patch("FOSRABack.src.processing.utils.processing_utils.config.EMBEDDING_MODEL_INSTANCE")
# @patch("FOSRABack.src.config.config.EMBEDDING_MODEL_INSTANCE")
# @patch("FOSRABack.src.processing.parse.selected_heavy_llm")
# @patch("FOSRABack.src.processing.parse.ChunkerManager")
# @patch("FOSRABack.src.processing.parse.Embedder")
# async def test_full_ingestion_pipeline(
#     mock_embedder_class,
#     mock_chunker_class,
#     mock_llm,
#     mock_embedding_instance_config,
#     mock_embedding_instance_utils,
#     mock_vector_service,
#     real_session: AsyncSession,
#     tmp_path: Path,
#     mock_source: Source,
# ):
#     test_file = tmp_path / "report.md"
#     _ = test_file.write_text("# Hello \nWorld")
#
#     # For test_full_ingestion_pipeline (markdown content)
#     actual_content_hash = generate_content_hash("# Hello \nWorld", 7)
#
#     # Create actual UserORM and WorkspaceORM instances
#     from FOSRABack.src.storage.models import UserORM, WorkspaceORM
#
#     test_user = UserORM(
#         user_id="user_123",
#         name="Test User",
#         enabled=True,
#         pages_limit=500,
#         pages_used=0,
#     )
#     real_session.add(test_user)
#     await real_session.commit()
#
#     test_workspace = WorkspaceORM(
#         workspace_id=7, name="Test Workspace", user_id="user_123"
#     )
#     real_session.add(test_workspace)
#     await real_session.commit()
#
#     # Setup embedding mock - must be AsyncMock with async methods
#     mock_embedding_instance = AsyncMock()
#     mock_embedding_instance.async_summary = AsyncMock(return_value=[0.1] * 384)
#     mock_embedding_instance.async_embed = AsyncMock(return_value=1)
#
#     # Assign the AsyncMock to both patched locations
#     mock_embedding_instance_config.return_value = mock_embedding_instance
#     mock_embedding_instance_config.async_summary = mock_embedding_instance.async_summary
#     mock_embedding_instance_config.async_embed = mock_embedding_instance.async_embed
#
#     mock_embedding_instance_utils.return_value = mock_embedding_instance
#     mock_embedding_instance_utils.async_summary = mock_embedding_instance.async_summary
#     mock_embedding_instance_utils.async_embed = mock_embedding_instance.async_embed
#
#     # Setup LLM mock - must support async invocation
#     mock_llm_instance = AsyncMock()
#     mock_llm_response = MagicMock()
#     mock_llm_response.content = "Test document summary"
#     mock_llm_instance.ainvoke = AsyncMock(return_value=mock_llm_response)
#     mock_llm.return_value = mock_llm_instance
#
#     # Setup chunker mock
#     c = Chunk(
#         chunk_id="01KAS3RK8K9M8F1HW6D2SVYDGD",
#         text="# Hello\nWorld",
#         source_hash=actual_content_hash,
#         start_index=0,
#         end_index=14,
#         token_count=5,
#         dense_vector=[0.1, 0.2, 0.3],
#         sparse_vector=SparseVector(indices=[1], values=[0.1]).model_dump(),
#         source_id="",
#     )
#     source_with_chunks = SourceWithRelations(
#         source_id="4321",
#         source_hash=actual_content_hash,
#         unique_id="uniq123",
#         content="# Hello\nWorld",
#         content_summary="summary",
#         workspace_ids=[7],
#         summary_embedding=str([0.1] * 384),
#         created_date=datetime.now(),
#         uploaded_at=datetime.now(),
#         chunks=[c],
#     )
#
#     mock_chunker_instance = AsyncMock()
#     mock_chunker_instance.chunk_single = AsyncMock(return_value=source_with_chunks)
#     mock_chunker_class.return_value = mock_chunker_instance
#
#     # Setup embedder instance mock
#     mock_embedder_instance = mock_embedding_instance
#     mock_embedder_class.return_value = mock_embedder_instance
#
#     mock_vector_service.store_vectors = AsyncMock()
#
#     # MAIN FUNC
#     await process_source_in_background(
#         file_path=str(test_file),
#         sourcename="report.md",
#         workspace_id=7,
#         user_id="user_123",
#         session=real_session,
#         task_logger=AsyncMock(),
#     )
#
#     # Verify vector storage was called
#     mock_vector_service.store_vectors.assert_called_once()
#
#     # Fix the query to properly join with origin
#     from FOSRABack.src.storage.models import OriginORM
#
#     result = await real_session.execute(
#         select(SourceORM).join(OriginORM).where(OriginORM.name == "report.md")
#     )
#
#     source = result.scalar_one()
#
#     # DATABASE STATE ASSERTIONS
#     result = await real_session.execute(
#         select(SourceORM).join(OriginORM).where(OriginORM.name == "report.md")
#     )
#     source = result.scalar_one()
#
#     # Origin assertions
#     assert source.origin.file_path == str(test_file)
#     assert source.origin.name == "report.md"
#     assert source.origin.source_type == "File"
#
#     # Source assertions
#     assert 7 in source.workspaces
#     assert source.source_hash is not None
#     assert source.unique_id is not None
#     assert source.content_summary is not None
#     assert source.summary_embedding is not None
#
#     # Chunk assertions
#     assert len(source.chunks) == 1
#     chunk: ChunkORM = source.chunks[0]
#     assert chunk.text is not None
#     assert chunk.source_hash == source.source_hash
#     assert chunk.token_count > 0
#
#     # Cleanup assertions
#     # assert not test_file.exists() #TODO: add this back later, clean up function was deleting test files
#
#     # Mock call assertions
#     mock_vector_service.store_vectors.assert_called_once()
#     call_args = mock_vector_service.store_vectors.call_args
#     assert len(call_args.kwargs["source_list"]) == 1
#
#
# @pytest.mark.asyncio
# @patch("FOSRABack.src.processing.parse.state.vector_service")
# @patch("FOSRABack.src.processing.utils.processing_utils.config.EMBEDDING_MODEL_INSTANCE")
# @patch("FOSRABack.src.config.config.EMBEDDING_MODEL_INSTANCE")
# @patch("FOSRABack.src.processing.parse.selected_heavy_llm")
# @patch("FOSRABack.src.processing.parse.ChunkerManager")
# @patch("FOSRABack.src.processing.parse.Embedder")
# @patch("FOSRABack.src.processing.utils.processing_utils.convert_document_to_markdown")
# @patch("FOSRABack.src.processing.parse.UnstructuredFileLoader")
# @patch("FOSRABack.src.config.config.ETL_SERVICE", "UNSTRUCTURED")
# async def test_ingestion_unstructured(
#     mock_loader_class,
#     mock_convert,
#     mock_embedder_class,
#     mock_chunker_class,
#     mock_llm,
#     mock_embedding_instance_config,
#     mock_embedding_instance_utils,
#     mock_vector_service,
#     real_session: AsyncSession,
#     tmp_path: Path,
#     mock_source: Source,
# ):
#     test_file = tmp_path / "report.pdf"
#     _ = test_file.write_text("fake pdf content")
#     from FOSRABack.src.processing.utils.processing_utils import generate_content_hash
#
#     actual_content_hash = generate_content_hash("parsed content markdown", 7)
#     # Create actual UserORM and WorkspaceORM instances
#     from FOSRABack.src.storage.models import UserORM, WorkspaceORM
#
#     # Check if user exists, if not create
#     stmt = select(UserORM).where(UserORM.user_id == "user_123")
#     result = await real_session.execute(stmt)
#     if not result.scalar_one_or_none():
#         test_user = UserORM(
#             user_id="user_123",
#             name="Test User",
#             enabled=True,
#             pages_limit=500,
#             pages_used=0,
#         )
#         real_session.add(test_user)
#
#         test_workspace = WorkspaceORM(id=7, name="Test Workspace", user_id="user_123")
#         real_session.add(test_workspace)
#         await real_session.commit()
#
#     # Setup embedding mock
#     mock_embedding_instance = AsyncMock()
#     mock_embedding_instance.async_summary = AsyncMock(return_value=[0.1] * 384)
#     mock_embedding_instance.async_embed = AsyncMock(return_value=1)
#
#     # Assign the AsyncMock to both patched locations
#     mock_embedding_instance_config.return_value = mock_embedding_instance
#     mock_embedding_instance_config.async_summary = mock_embedding_instance.async_summary
#     mock_embedding_instance_config.async_embed = mock_embedding_instance.async_embed
#
#     mock_embedding_instance_utils.return_value = mock_embedding_instance
#     mock_embedding_instance_utils.async_summary = mock_embedding_instance.async_summary
#     mock_embedding_instance_utils.async_embed = mock_embedding_instance.async_embed
#
#     # Setup LLM mock
#     mock_llm_instance = AsyncMock()
#     mock_llm_response = MagicMock()
#     mock_llm_response.content = "Test document summary from unstructured"
#     mock_llm_instance.ainvoke = AsyncMock(return_value=mock_llm_response)
#     mock_llm.return_value = mock_llm_instance
#
#     # Setup UnstructuredFileLoader mock
#     mock_loader_instance = AsyncMock()
#     mock_doc = MagicMock()
#     mock_doc.page_content = "parsed content"
#     mock_loader_instance.aload = AsyncMock(return_value=[mock_doc])
#     mock_loader_class.return_value = mock_loader_instance
#
#     # Setup convert_document_to_markdown mock
#     mock_convert.return_value = "parsed content markdown"
#
#     # Setup chunker mock
#     fake_chunk = Chunk(
#         chunk_id="01KAS3RK8K9M8F1HW6D2SVYDGE",
#         text="parsed content",
#         source_hash=actual_content_hash,
#         start_index=0,
#         end_index=14,
#         token_count=5,
#         dense_vector=[0.1, 0.2, 0.3],
#         sparse_vector=SparseVector(indices=[1], values=[0.1]).model_dump(),
#         source_id="",
#     )
#
#     mock_source_with_chunks = SourceWithRelations(
#         source_id="test_source_id",
#         source_hash="hash123",
#         unique_id="unique_test",
#         content="parsed content markdown",
#         content_summary="Test summary",
#         workspace_ids=[7],
#         summary_embedding=str([0.1] * 384),
#         created_date=datetime.now(),
#         uploaded_at=datetime.now(),
#         chunks=[fake_chunk],
#     )
#
#     mock_chunker_instance = AsyncMock()
#     mock_chunker_instance.chunk_single = AsyncMock(return_value=mock_source_with_chunks)
#     mock_chunker_class.return_value = mock_chunker_instance
#
#     # Setup embedder instance mock
#     mock_embedder_instance = mock_embedding_instance
#     mock_embedder_class.return_value = mock_embedder_instance
#
#     mock_vector_service.store_vectors = AsyncMock()
#
#     await process_source_in_background(
#         file_path=str(test_file),
#         sourcename="report.pdf",
#         workspace_id=7,
#         user_id="user_123",
#         session=real_session,
#         task_logger=AsyncMock(),
#     )
#
#     # MOCK BEHAVIOR ASSERTIONS
#     mock_loader_class.assert_called_once()
#     loader_call = mock_loader_class.call_args
#     assert loader_call.kwargs["file_path"] == str(test_file)
#     assert loader_call.kwargs["mode"] == "elements"
#     assert loader_call.kwargs["strategy"] == "auto"
#
#     mock_loader_instance.aload.assert_called_once()
#     mock_convert.assert_called_once()
#
#     # Verify conversion was called with docs
#     convert_call_args = mock_convert.call_args
#     assert len(convert_call_args[0][0]) == 1  # One document
#
#     # DATABASE STATE ASSERTIONS
#     from FOSRABack.src.storage.models import OriginORM
#
#     result = await real_session.execute(
#         select(SourceORM).join(OriginORM).where(OriginORM.name == "report.pdf")
#     )
#     source = result.scalar_one()
#
#     assert source.origin.name == "report.pdf"
#     assert 7 in source.workspaces
#     assert len(source.chunks) > 0
#     assert source.content_summary is not None
#
#     # VECTOR STORAGE ASSERTIONS
#     mock_vector_service.store_vectors.assert_called_once()
#     vector_call = mock_vector_service.store_vectors.call_args
#     stored_sources = vector_call.kwargs["source_list"]
#     assert len(stored_sources) == 1
#     assert stored_sources[0].source_id == source.source_id
#
#     # PAGE LIMIT ASSERTIONS
#     user_result = await real_session.execute(
#         select(UserORM).where(UserORM.user_id == "user_123")
#     )
#     user = user_result.scalar_one()
#     assert user.pages_used >= 0  # Pages were tracked
#
#
# @pytest.mark.asyncio
# @patch("FOSRABack.src.processing.parse.state.vector_service")
# @patch("FOSRABack.src.processing.utils.processing_utils.config.EMBEDDING_MODEL_INSTANCE")
# @patch("FOSRABack.src.config.config.EMBEDDING_MODEL_INSTANCE")
# @patch("FOSRABack.src.processing.parse.selected_heavy_llm")
# @patch("FOSRABack.src.processing.parse.ChunkerManager")
# @patch("FOSRABack.src.processing.parse.Embedder")
# @patch("FOSRABack.src.convo.utils.docling_helper.DoclingService")
# @patch("FOSRABack.src.config.config.ETL_SERVICE", "DOCLING")
# async def test_ingestion_docling(
#     mock_docling_service_class,
#     mock_embedder_class,
#     mock_chunker_class,
#     mock_llm,
#     mock_embedding_instance_config,
#     mock_embedding_instance_utils,
#     mock_vector_service,
#     real_session: AsyncSession,
#     tmp_path: Path,
# ):
#     test_file = tmp_path / "doc.pdf"
#     _ = test_file.write_text("fake pdf content")
#
#     from FOSRABack.src.processing.utils.processing_utils import generate_content_hash
#
#     actual_content_hash = generate_content_hash("docling content", 7)
#     # Ensure user/workspace exist
#     from FOSRABack.src.storage.models import UserORM, WorkspaceORM
#
#     stmt = select(UserORM).where(UserORM.user_id == "user_123")
#     result = await real_session.execute(stmt)
#     if not result.scalar_one_or_none():
#         test_user = UserORM(
#             user_id="user_123",
#             name="Test User",
#             enabled=True,
#             pages_limit=500,
#             pages_used=0,
#         )
#         real_session.add(test_user)
#         test_workspace = WorkspaceORM(id=7, name="Test Workspace", user_id="user_123")
#         real_session.add(test_workspace)
#         await real_session.commit()
#
#         # Setup embedding mock
#     mock_embedding_instance = AsyncMock()
#     mock_embedding_instance.async_summary = AsyncMock(return_value=[0.1] * 384)
#     mock_embedding_instance.async_embed = AsyncMock(return_value=1)
#
#     # Assign the AsyncMock to both patched locations
#     mock_embedding_instance_config.return_value = mock_embedding_instance
#     mock_embedding_instance_config.async_summary = mock_embedding_instance.async_summary
#     mock_embedding_instance_config.async_embed = mock_embedding_instance.async_embed
#
#     mock_embedding_instance_utils.return_value = mock_embedding_instance
#     mock_embedding_instance_utils.async_summary = mock_embedding_instance.async_summary
#     mock_embedding_instance_utils.async_embed = mock_embedding_instance.async_embed
#
#     # Setup LLM mock
#     mock_llm_instance = AsyncMock()
#     mock_llm_response = MagicMock()
#     mock_llm_response.content = "Test document summary from docling"
#     mock_llm_instance.ainvoke = AsyncMock(return_value=mock_llm_response)
#     mock_llm.return_value = mock_llm_instance
#
#     # Setup Docling Service mock
#     mock_docling_service = AsyncMock()
#     mock_docling_service.process_document = AsyncMock(
#         return_value={"content": "docling content"}
#     )
#     mock_docling_service.process_large_document_summary = AsyncMock(
#         return_value="docling summary"
#     )
#     mock_docling_service_class.return_value = mock_docling_service
#
#     # Setup chunker mock
#     fake_chunk = Chunk(
#         chunk_id="01KAS3RK8K9M8F1HW6D2SVYDGF",
#         text="docling content",
#         source_hash=actual_content_hash,
#         start_index=0,
#         end_index=14,
#         token_count=5,
#         dense_vector=[0.1, 0.2, 0.3],
#         sparse_vector=SparseVector(indices=[1], values=[0.1]).model_dump(),
#         source_id="",
#     )
#
#     mock_source_with_chunks = SourceWithRelations(
#         source_id="test_docling_source",
#         source_hash=actual_content_hash,
#         unique_id="unique_docling",
#         content="docling content",
#         content_summary="Docling test summary",
#         workspace_ids=[7],
#         summary_embedding=str([0.1] * 384),
#         created_date=datetime.now(),
#         uploaded_at=datetime.now(),
#         chunks=[fake_chunk],
#     )
#
#     mock_chunker_instance = AsyncMock()
#     mock_chunker_instance.chunk_single = AsyncMock(return_value=mock_source_with_chunks)
#     mock_chunker_class.return_value = mock_chunker_instance
#
#     # Setup embedder instance mock
#     mock_embedder_instance = mock_embedding_instance
#     mock_embedder_class.return_value = mock_embedder_instance
#
#     mock_vector_service.store_vectors = AsyncMock()
#
#     await process_source_in_background(
#         file_path=str(test_file),
#         sourcename="doc.pdf",
#         workspace_id=7,
#         user_id="user_123",
#         session=real_session,
#         task_logger=AsyncMock(),
#     )
#
#     # DOCLING SERVICE ASSERTIONS
#     mock_docling_service.process_document.assert_called_once()
#     docling_call = mock_docling_service.process_document.call_args
#     assert docling_call[0][0] == str(test_file)  # file_path
#     assert docling_call[0][1] == "doc.pdf"  # name
#
#     # DATABASE STATE ASSERTIONS
#     from FOSRABack.src.storage.models import OriginORM
#
#     result = await real_session.execute(
#         select(SourceORM).join(OriginORM).where(OriginORM.name == "doc.pdf")
#     )
#     source = result.scalar_one()
#
#     assert source.origin.name == "doc.pdf"
#     assert 7 in source.workspaces
#     assert source.content_summary is not None
#     assert "# DOCUMENT METADATA" in source.content_summary  # Docling-specific
#     assert "# DOCUMENT SUMMARY" in source.content_summary
#
#     # VECTOR STORAGE ASSERTIONS
#     mock_vector_service.store_vectors.assert_called_once()
#
#     # PAGE USAGE ASSERTIONS
#     user_result = await real_session.execute(
#         select(UserORM).where(UserORM.user_id == "user_123")
#     )
#     user = user_result.scalar_one()
#     assert user.pages_used > 0  # Docling should have updated this
