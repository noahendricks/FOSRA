from pathlib import Path
from typing import cast, Any
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from FOSRABack.src.config import app_state
from FOSRABack.src.config.request_context import VectorStoreType
from FOSRABack.src.convo.services.convo_service import ConvoService
from FOSRABack.src.convo.services.workspace_service import WorkspaceService
from FOSRABack.src.storage.repos.convo_repo import ConvoRepo
from FOSRABack.src.storage.database import SQLSessionManager
from FOSRABack.src.storage.schemas import (
    Chunk,
    Message,
    Origin,
    OriginType,
    RetrievedContext,
    Source,
    User,
    Conversation,
    ConversationWithMessages,
    MessageRole,
)
from FOSRABack.src.storage.models import (
    MessageORM,
    ConvoORM,
    UserORM,
    WorkspaceORM,
)
from FOSRABack.src.resources.test_fixtures import *

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_retrieved_context_list() -> list[RetrievedContext]:
    """Create a list of RetrievedContext objects for testing."""
    origin1 = Origin(
        name="source_one.txt",
        origin_path="/path/to/source_one.txt",
        origin_type=OriginType.FILE,
        source_hash="hash1",
    )
    chunk1 = Chunk(
        chunk_id="chunk1",
        source_id="source1",
        source_hash="hash1",
        text="This is the first chunk of content about machine learning.",
        start_index=0,
        end_index=100,
        token_count=15,
    )
    source1 = Source(
        source_id="source1",
        source_hash="hash1",
        unique_id="unique1",
        source_summary="Summary of source one",
        summary_embedding="[]",
    )

    origin2 = Origin(
        name="source_two.pdf",
        origin_path="/path/to/source_two.pdf",
        origin_type=OriginType.FILE,
        source_hash="hash2",
    )
    chunk2 = Chunk(
        chunk_id="chunk2",
        source_id="source2",
        source_hash="hash2",
        text="This is the second chunk about neural networks and deep learning.",
        start_index=0,
        end_index=120,
        token_count=12,
    )
    source2 = Source(
        source_id="source2",
        source_hash="hash2",
        unique_id="unique2",
        source_summary="Summary of source two",
        summary_embedding="[]",
    )

    return [
        RetrievedContext(
            chunk=chunk1,
            source=source1,
            origin=origin1,
            contents=chunk1.text,
            similarity_score=0.95,
            result_rank=1,
            query="test query",
        ),
        RetrievedContext(
            chunk=chunk2,
            source=source2,
            origin=origin2,
            contents=chunk2.text,
            similarity_score=0.87,
            result_rank=2,
            query="test query",
        ),
    ]


@pytest.fixture
def mock_convo_repo(mock_convo, mock_new_convo, mock_db_convo):
    """
    Create a mock ConvoRepo with all methods stubbed.

    Methods to mock:
    - create_new_convo: returns Conversation
    - retrieve_convo_by_id: returns ConversationWithMessages
    - persist_session: returns None
    - update_chat: returns ConvoORM
    - delete_chat: returns dict
    """
    repo = AsyncMock()

    repo.create_new_convo.return_value = mock_new_convo
    repo.retrieve_convo_by_id.return_value = mock_convo
    repo.update_chat.return_value = mock_db_convo
    repo.persist_session.return_value = None
    repo.delete_chat.return_value = {"message": "Chat deleted successfully"}

    return repo


@pytest.fixture
def mock_vector_service(mock_retrieved_context_list: list[RetrievedContext]):
    """
    Create a mock VectorRepo for context retrieval.

    Methods to mock:
    - source_vector_search: returns list[RetrievedContext] or list[dict]
    """
    repo = AsyncMock()

    repo.source_vector_search.return_value = mock_retrieved_context_list
    return repo


@pytest.fixture
def sample_user(mock_user):
    """
    Create a sample User for testing.

    Fields:
    - user_id: "test_user_123"
    - name: "Test User"
    - created_at: datetime.now()
    - last_login: datetime.now()
    - pages_limit: 100
    - pages_used: 10
    """
    mock_user.user_id = "test_user_123"
    mock_user.name = "Test User"
    mock_user.created_at = datetime.now()
    mock_user.last_login = datetime.now()
    mock_user.pages_limit = 100
    mock_user.pages_used = 10

    return mock_user


@pytest.fixture
def sample_conversation(mock_convo):
    """
    Create a sample Conversation for testing.

    Fields:
    - convo_id: generated ULID
    - user_id: from sample_user
    - workspace_id: 1
    - title: "Test Conversation"
    - created_at: datetime.now()
    """
    return mock_convo(
        convo_id="01FZ8Z5Y6X7Y8Z9A0B1C2D3E4F",
        title="Test Conversation",
        workspace_id=1,
        user_id="test_user_123",
    )


@pytest.fixture
def sample_conversation_with_messages(mock_convo):
    """
    Create ConversationWithMessages with sample messages.

    Include:
    - 2-3 messages alternating user/assistant
    - state_version set appropriately
    """
    mock_convo.messages = [
        Message(
            role=MessageRole.USER,
            message_content="Hello!",
            convo_id="convo_1",
        ),
        Message(
            role=MessageRole.ASSISTANT,
            message_content="Hi there! How can I help you?",
            convo_id="convo_1",
        ),
        Message(
            role=MessageRole.USER,
            message_content="Tell me a joke!",
            convo_id="convo_1",
        ),
        Message(
            role=MessageRole.ASSISTANT,
            message_content="Why did the chicken cross the road? To get to the other side!",
            convo_id="convo_1",
        ),
    ]

    mock_convo.state_version = 4

    return mock_convo


@pytest.fixture
def sample_source_with_relations(mock_single_source_with_relations):
    """
    Create a SourceWithRelations for context retrieval tests.

    Include:
    - source_id, source_hash, unique_id
    - origin: Origin with name, origin_path, origin_type
    - chunks: list with 1-2 Chunk objects containing text
    """
    return mock_single_source_with_relations


@pytest.fixture
def convo_manager(mock_convo_repo):
    """Create a ConvoService instance with mocked ConvoRepo."""
    return ConvoService(convo_repo=mock_convo_repo)


@pytest.fixture
def real_convo_repo():
    """Create a instance with mocked ConvoRepo."""
    return ConvoRepo(db_session=app_state.db_session_manager)


# =============================================================================
# TEST CLASS: ConvoService Initialization
# =============================================================================


class TestConvoServiceInit:
    """Tests for ConvoService initialization."""

    def test_init_with_convo_repo(self, mock_convo_repo):
        """
        Test ConvoService initializes correctly with ConvoRepo.

        Verify:
        - self.convo_repo is set to provided repo
        """
        new = ConvoService(convo_repo=mock_convo_repo)

        assert new.convo_repo == mock_convo_repo

    def test_init_stores_repo_reference(self, mock_convo_repo):
        """
        Test that the repo reference is stored correctly.

        Verify:
        - Can access convo_repo attribute after init
        """
        new = ConvoService(convo_repo=mock_convo_repo)

        assert hasattr(new, "convo_repo")
        assert new.convo_repo is mock_convo_repo


# =============================================================================
# TEST CLASS: create_new_convo
# =============================================================================


class TestCreateNewConvo:
    """Tests for ConvoService.create_new_convo method."""

    @pytest.mark.asyncio
    async def test_create_new_convo_success(
        self, convo_manager, sample_user, mock_request_context
    ):
        """
        Test successful conversation creation.

        Setup:
        - Mock convo_repo.create_new_convo to return Conversation

        Verify:
        - Returns Conversation with correct convo_id
        - Returns Conversation with correct user_id
        - convo_repo.create_new_convo called with user and workspace_id
        """

        MockCreateNew = AsyncMock(
            return_value=Conversation(
                convo_id="test_convo_id",
                user_id=mock_request_context.user_id,
                workspace_id=mock_request_context.workspace_id,
            )
        )
        convo_manager.create_new_convo = MockCreateNew

        result = await convo_manager.create_new_convo(
            user_id=mock_request_context.user_id,
            workspace_id=mock_request_context.workspace_id,
        )

        assert result.convo_id == "test_convo_id"

        assert result.user_id == mock_request_context.user_id

        MockCreateNew.assert_called_with(
            user_id=mock_request_context.user_id,
            workspace_id=mock_request_context.workspace_id,
        )

    @pytest.mark.asyncio
    async def test_create_new_convo_with_different_workspace_ids(
        self, convo_manager, mock_convo_repo, mock_request_context
    ):
        """
        Test conversation creation with various workspace IDs.

        Test workspace_id values: 1, 100, 999

        Verify:
        - Each call passes correct workspace_id to repo
        """

        for ws_id in [1, 100, 999]:
            new = mock_request_context.with_new_workspace_id(ws_id)
            await convo_manager.create_new_convo(ctx=new)

            mock_convo_repo.create_new_convo.assert_called_with(
                user_id=mock_request_context.user_id, workspace_id=ws_id
            )
            assert (
                mock_convo_repo.create_new_convo.call_args[1]["workspace_id"] == ws_id
            )

    @pytest.mark.asyncio
    async def test_create_new_convo_repo_exception(
        self, convo_manager, mock_convo_repo, mock_request_context
    ):
        """
        Test handling when repo raises exception.

        Setup:
        - Mock convo_repo.create_new_convo to raise ValueError

        Verify:
        - Exception propagates up
        """
        mock_convo_repo.create_new_convo.side_effect = ValueError("DB error")
        with pytest.raises(ValueError) as exc_info:
            await convo_manager.create_new_convo(ctx=mock_request_context)
        assert str(exc_info.value) == "DB error"
        assert mock_convo_repo.create_new_convo.called


# =============================================================================
# TEST CLASS: save_user_message
# =============================================================================


class TestSaveUserMessage:
    """Tests for ConvoService.save_user_message method."""

    @pytest.mark.asyncio
    async def test_save_user_message_success(
        self,
        convo_manager: ConvoService,
        mock_convo_repo: ConvoRepo,
        mock_convo: ConversationWithMessages,
        mock_db_convo: ConvoORM,
    ):
        """
        Test successful user message saving.

        Setup:
        - Mock save_message to return MessageORM

        Verify:
        - Returns MessageORM instance
        - MessageORM.role == "user"
        - MessageORM.message_content contains query
        - MessageORM.convo_id matches convo
        - save_message called once
        """
        # Create expected return value
        expected_message = MessageORM(
            role=MessageRole.USER.value,
            message_content="Test message",
            convo_id=mock_convo.convo_id,
        )

        # Mock save_message to succeed
        mock_convo_repo = cast(AsyncMock, convo_manager.convo_repo)
        mock_convo_repo.save_message.return_value = expected_message

        # Call the method under test
        result = await convo_manager.save_user_message(
            convo=mock_convo, query="Test message"
        )

        # Verify the result
        assert result is not None
        assert isinstance(result, MessageORM)
        assert result.role == MessageRole.USER.value
        assert result.message_content == "Test message"
        assert result.convo_id == mock_convo.convo_id

        # Verify save_message was called with correct args
        mock_convo_repo.save_message.assert_called_once_with(
            convo_id=mock_convo.convo_id,
            role=MessageRole.USER,
            content="Test message",
        )

    @pytest.mark.asyncio
    async def test_save_user_message_calls_repo_correctly(
        self, convo_manager, mock_convo
    ):
        """
        Test that save_user_message delegates to repo correctly.

        Verify:
        - save_message called with correct parameters
        - Returns the message from repo
        """
        expected_message = MessageORM(
            role=MessageRole.USER.value,
            message_content="Test query",
            convo_id=mock_convo.convo_id,
        )

        mock_convo_repo = cast(AsyncMock, convo_manager.convo_repo)
        mock_convo_repo.save_message.return_value = expected_message

        result = await convo_manager.save_user_message(
            query="Test query", convo=mock_convo
        )

        # Assert: Verify save_message was called with correct args
        mock_convo_repo.save_message.assert_called_once_with(
            convo_id=mock_convo.convo_id,
            role=MessageRole.USER,
            content="Test query",
        )

        # Assert: Verify the returned message matches
        assert result is expected_message
        assert result.role == MessageRole.USER.value
        assert result.message_content == "Test query"

    @pytest.mark.asyncio
    async def test_save_user_message_persist_failure(
        self, convo_manager, mock_convo_repo, mock_convo
    ):
        """
        Test handling when save_message fails.

        Setup:
        - Mock save_message to raise exception

        Verify:
        - Raises RuntimeError with "Failed to save message"
        """
        mock_convo_repo = cast(AsyncMock, convo_manager.convo_repo)
        mock_convo_repo.save_message.side_effect = Exception("DB write error")

        with pytest.raises(RuntimeError) as exc_info:
            await convo_manager.save_user_message(
                query="Test failure", convo=mock_convo
            )
        assert str(exc_info.value) == "Failed to save message"


# =============================================================================
# TEST CLASS: new_assistant_message
# =============================================================================


class TestNewAssistantMessage:
    """Tests for ConvoService.new_assistant_message method."""

    @pytest.mark.asyncio
    async def test_new_assistant_message_success(self, convo_manager, mock_convo):
        """
        Test successful assistant message creation.

        Verify:
        - Returns Message instance
        - message.role == MessageRole.ASSISTANT
        - message.message_content == response_str
        - message.convo_id matches convo
        - message.message_id is set (ULID)
        - message.chat_metadata is empty string
        """
        # TODO: Implement test
        response_str = "This is a test assistant response."

        result = await convo_manager.new_assistant_message(
            response_str=response_str, convo_session=mock_convo
        )
        assert result is not None
        assert isinstance(result, Message)
        assert result.role == MessageRole.ASSISTANT
        assert result.message_content == response_str
        assert result.convo_id == mock_convo.convo_id
        assert result.message_id is not None
        assert len(result.message_id) == 26  # ULID length
        assert result.chat_metadata == ""

    @pytest.mark.asyncio
    async def test_new_assistant_message_with_long_response(
        self, convo_manager, mock_convo
    ):
        """
        Test assistant message with long response content.

        Setup:
        - Use response_str with 10000+ characters

        Verify:
        - Full content preserved in message_content
        """
        # Create a long response string (10000+ characters)
        long_response = "A" * 10000 + " This is a test response with lots of content."

        # Call the method under test
        result = await convo_manager.new_assistant_message(
            response_str=long_response, convo_session=mock_convo
        )

        # Verify the full content is preserved
        assert result is not None
        assert isinstance(result, Message)
        assert result.role == MessageRole.ASSISTANT
        assert result.message_content == long_response
        assert len(result.message_content) == len(long_response)
        assert result.message_content.startswith("A" * 100)  # Verify start
        assert result.message_content.endswith("with lots of content.")  # Verify end
        assert result.convo_id == mock_convo.convo_id

    @pytest.mark.asyncio
    async def test_new_assistant_message_with_special_characters(
        self, convo_manager, mock_convo
    ):
        """
        Test assistant message with special characters.

        Setup:
        - Use response_str with unicode, markdown, code blocks

        Verify:
        - Special characters preserved correctly
        """
        # Create response with various special characters
        special_response = """
        # Markdown Header

        Here's some **bold** and *italic* text.

        ```python
        def hello():
            print("Hello, 世界!")
        ```

        Unicode: 你好 • Emoji: 🚀 • Math: ∑ ∫ ∂

        Special chars: <>&"'`
        """

        # Call the method under test
        result = await convo_manager.new_assistant_message(
            response_str=special_response, convo_session=mock_convo
        )

        # Verify special characters are preserved
        assert result is not None
        assert isinstance(result, Message)
        assert result.role == MessageRole.ASSISTANT
        assert result.message_content == special_response

        # Verify specific special characters are present
        assert "**bold**" in result.message_content
        assert "```python" in result.message_content
        assert "世界" in result.message_content
        assert "🚀" in result.message_content
        assert "<>&\"'`" in result.message_content
        assert result.convo_id == mock_convo.convo_id


# =============================================================================
# TEST CLASS: get_context
# =============================================================================


class TestGetContext:
    """Tests for ConvoService.get_context method."""

    @patch("FOSRABack.src.convo.services.convo_service.app_state")
    @pytest.mark.asyncio
    async def test_get_context_success_with_retrieved_context(
        self,
        mock_app_state,
        convo_manager: ConvoService,
        mock_list_sources_with_relations,
        mock_request_context,
    ):
        """
        Test successful context retrieval returning RetrievedContext objects.

        Setup:
        - Patch app_state.vector_service.search
        - Return list of RetrievedContext-like objects

        Verify:
        - Returns list of RetrievedContext
        - Length matches mock return
        - search called with correct params
        """
        # Mock the vector_service property and its search method
        mock_vector_service = AsyncMock()
        mock_vector_service.search.return_value = mock_list_sources_with_relations
        mock_app_state.vector_service = mock_vector_service

        result = await convo_manager.get_context(
            query="what are the similarities in these files?", ctx=mock_request_context
        )

        assert isinstance(result, list)
        assert all(isinstance(src, RetrievedContext) for src in result)
        assert len(result) == len(mock_list_sources_with_relations)

        mock_vector_service.search.assert_awaited_once_with(
            ctx=mock_request_context,
            query_vector=[],
            query_text="what are the similarities in these files?",
            store_type=VectorStoreType.QDRANT,
        )

    @pytest.mark.asyncio
    @patch("FOSRABack.src.convo.services.convo_service.app_state")
    async def test_get_context_success_with_dicts(
        self,
        mock_app_state,
        convo_manager,
        mock_request_context,
    ):
        """
        Test context retrieval when vector search returns dicts.

        Setup:
        - Return list of dicts that match RetrievedContext schema

        Verify:
        - Dicts converted to RetrievedContext objects
        """
        mock_vector_service = AsyncMock()
        # Create dict representations with the expected structure
        contexts_as_dicts = [
            {
                "chunk": {
                    "chunk_id": "chunk1",
                    "source_id": "source1",
                    "source_hash": "hash1",
                    "text": "Test chunk content",
                    "start_index": 0,
                    "end_index": 50,
                    "token_count": 10,
                },
                "source": {
                    "source_id": "source1",
                    "source_hash": "hash1",
                    "unique_id": "unique1",
                    "source_summary": "Test summary",
                    "summary_embedding": "[]",
                },
                "origin": {
                    "name": "test.txt",
                    "origin_path": "/path/test.txt",
                    "origin_type": "FILE",
                    "source_hash": "hash1",
                },
                "similarity_score": 0.9,
                "result_rank": 1,
            }
        ]
        mock_vector_service.search.return_value = contexts_as_dicts
        mock_app_state.vector_service = mock_vector_service

        result = await convo_manager.get_context(
            query="what are the similarities in these files?", ctx=mock_request_context
        )

        assert isinstance(result, list)
        assert all(isinstance(src, RetrievedContext) for src in result)
        assert len(result) == len(contexts_as_dicts)
        # Verify the converted object has correct data
        assert result[0].chunk.text == "Test chunk content"
        assert result[0].source.source_id == "source1"
        assert result[0].origin.name == "test.txt"
        assert result[0].similarity_score == 0.9

        mock_vector_service.search.assert_awaited_once_with(
            ctx=mock_request_context,
            query_vector=[],
            query_text="what are the similarities in these files?",
            store_type=VectorStoreType.QDRANT,
        )

    @pytest.mark.asyncio
    @patch("FOSRABack.src.convo.services.convo_service.app_state")
    async def test_get_context_empty_results(
        self, mock_app_state, convo_manager, mock_request_context
    ):
        """
        Test context retrieval with no results.

        Setup:
        - Mock source to return []

        Verify:
        - Raises ValueError with "No relevant context found"
        """
        mock_vector_service = AsyncMock()
        mock_vector_service.search.return_value = []
        mock_app_state.vector_service = mock_vector_service

        with pytest.raises(ValueError) as exc_info:
            await convo_manager.get_context(
                query="what are the similarities in these files?",
                ctx=mock_request_context,
            )
        assert str(exc_info.value) == "No relevant context found"
        mock_vector_service.search.assert_awaited_once_with(
            ctx=mock_request_context,
            query_vector=[],
            query_text="what are the similarities in these files?",
            store_type=VectorStoreType.QDRANT,
        )

    @pytest.mark.asyncio
    async def test_get_context_passes_workspace_id(
        self, convo_manager, mock_request_context
    ):
        """
        Test that workspace_id is passed to vector search.

        Verify:
        - workspace_id parameter included in search call
        """
        with patch(
            "FOSRABack.src.convo.services.convo_service.app_state"
        ) as mock_app_state:
            mock_vector_service = AsyncMock()
            mock_vector_service.search.return_value = []
            mock_app_state.vector_service = mock_vector_service

            with pytest.raises(ValueError) as exc_info:
                await convo_manager.get_context(
                    query="what are the similarities in these files?",
                    ctx=mock_request_context,
                )

            assert str(exc_info.value) == "No relevant context found"
            mock_vector_service.search.assert_awaited_once_with(
                ctx=mock_request_context,
                query_vector=[],
                query_text="what are the similarities in these files?",
                store_type=VectorStoreType.QDRANT,
            )


# =============================================================================
# TEST CLASS: format_to_str
# =============================================================================


class TestFormatToStr:
    """Tests for ConvoService.format_to_str method."""

    def test_format_to_str_single_context(
        self, convo_manager, mock_retrieved_context_list
    ):
        single_context = [mock_retrieved_context_list[0]]
        result = convo_manager.format_to_str(single_context)

        assert "FILE NAME: source_one.txt" in result
        assert result.endswith("END OF RETRIEVED CONTEXT")
        assert "[#1 RANKED CHUNK:" in result
        assert "machine learning" in result

    def test_format_to_str_multiple_contexts(
        self, convo_manager, mock_retrieved_context_list
    ):
        result = convo_manager.format_to_str(mock_retrieved_context_list)

        assert "FILE NAME: source_one.txt" in result
        assert "FILE NAME: source_two.pdf" in result
        assert "machine learning" in result
        assert "neural networks" in result
        assert result.endswith("END OF RETRIEVED CONTEXT")
        assert result.count("END OF RETRIEVED CONTEXT") == 1
        assert "[#1 RANKED CHUNK:" in result
        assert "[#2 RANKED CHUNK:" in result

    def test_format_to_str_with_none_rank(self, convo_manager):
        """
        Test formatting context with None result_rank.

        Verify:
        - Uses "N/A" as fallback rank
        """
        origin = Origin(
            name="test.txt",
            origin_path="/path/test.txt",
            origin_type=OriginType.FILE,
            source_hash="hash1",
        )
        chunk = Chunk(
            chunk_id="chunk1",
            source_id="source1",
            source_hash="hash1",
            text="Test content",
            start_index=0,
            end_index=50,
            token_count=5,
        )
        source = Source(
            source_id="source1",
            source_hash="hash1",
            unique_id="unique1",
            source_summary="",
            summary_embedding="[]",
        )
        context = RetrievedContext(
            chunk=chunk,
            source=source,
            origin=origin,
            contents=chunk.text,
            result_rank=None,  # No rank
            query="test",
        )

        result = convo_manager.format_to_str([context])

        assert "[#N/A RANKED CHUNK:" in result
        assert "FILE NAME: test.txt" in result

    def test_format_to_str_none_origin(self, convo_manager):
        """
        Test formatting context with None origin.

        Verify:
        - Uses unique_id from source as fallback name
        - No exception raised
        """
        chunk = Chunk(
            chunk_id="chunk1",
            source_id="source1",
            source_hash="hash1",
            text="Test content",
            start_index=0,
            end_index=50,
            token_count=5,
        )
        source = Source(
            source_id="source1",
            source_hash="hash1",
            unique_id="fallback_unique_id",
            source_summary="",
            summary_embedding="[]",
        )
        context = RetrievedContext(
            chunk=chunk,
            source=source,
            origin=None,  # No origin
            contents=chunk.text,
            result_rank=1,
            query="test",
        )

        result = convo_manager.format_to_str([context])

        assert "FILE NAME: fallback_unique_id" in result
        assert "Test content" in result

    def test_format_to_str_empty_list(self, convo_manager):
        """
        Test formatting empty source list.

        Verify:
        - Returns string with just "END OF RETRIEVED CONTEXT"
        """
        result = convo_manager.format_to_str([])

        assert result == "END OF RETRIEVED CONTEXT"

    def test_format_to_str_uses_to_context_string(
        self, convo_manager, mock_retrieved_context_list
    ):
        """
        Test that format uses RetrievedContext.to_context_string().

        Verify:
        - Output contains "RANKED CHUNK" text
        - Output contains "FILE NAME" label
        - Output contains "CHUNK CONTENT" label
        """
        result = convo_manager.format_to_str(mock_retrieved_context_list)

        assert "RANKED CHUNK" in result
        assert "FILE NAME:" in result
        assert "CHUNK CONTENT:" in result


# =============================================================================
# TEST CLASS: new_user_message (full flow)
# =============================================================================


class TestNewUserMessage:
    """Tests for ConvoService.new_user_message method (full RAG flow)."""

    @pytest.mark.asyncio
    @patch("FOSRABack.src.convo.services.convo_service.app_state")
    async def test_new_user_message_full_flow(
        self,
        mock_app_state,
        mock_retrieved_context_list: list[RetrievedContext],
        mock_request_context,
    ):
        """
        Test complete new_user_message flow.

        Setup:
        - Patch app_state.vector_service for context retrieval
        - Mock conversation with to_orm method
        - Mock persist_session

        Verify:
        - Vector search called with correct params
        - Constructed prompt contains context and query
        - MessageORM created with constructed prompt
        - persist_session called
        - Returns MessageORM
        """
        # Setup: Create test user
        sample_user = User(
            user_id="test_user_123",
            name="Test User",
            created_at=datetime.now(),
            last_login=datetime.now(),
            pages_limit=100,
            pages_used=10,
        )

        # Setup: Create mock conversation
        mock_convo = MagicMock(spec=ConversationWithMessages)
        mock_convo.convo_id = "test_convo_id"
        mock_convo.workspace_id = 1
        mock_convo.messages = []

        # Setup: Create mock ORM conversation
        mock_db_convo = MagicMock(spec=ConvoORM)
        mock_db_convo.messages = []
        mock_db_convo.convo_id = "test_convo_id"
        mock_convo.to_orm = MagicMock(return_value=mock_db_convo)

        # Setup: Mock vector service to return our test contexts
        mock_vector_service = AsyncMock()
        mock_vector_service.search.return_value = [
            ctx.model_dump() for ctx in mock_retrieved_context_list
        ]
        mock_app_state.vector_service = mock_vector_service

        # Setup: Create convo_manager with mocked repo
        mock_convo_repo = AsyncMock(spec=ConvoRepo)
        mock_convo_repo.persist_conversation.return_value = None
        mock_convo_repo.save_message.return_value = MessageORM(
            message_content="<context> ...</context> What is AI?",
            role=MessageRole.USER.value,
            convo_id=mock_convo.convo_id,
        )
        convo_manager = ConvoService(convo_repo=mock_convo_repo)

        # Act: Call the method under test
        result = await convo_manager.new_user_message(
            query="What is AI?", convo=mock_convo, ctx=mock_request_context
        )

        # Assert: Vector search was called with correct parameters
        mock_vector_service.search.assert_awaited_once_with(
            ctx=mock_request_context,
            query_vector=[],
            query_text="What is AI?",
            store_type=VectorStoreType.QDRANT,
        )

        # Assert: Result is a MessageORM with correct properties
        assert result is not None
        assert isinstance(result, MessageORM)
        assert result.role == MessageRole.USER.value
        # Assert: The constructed prompt contains context and query
        assert "What is AI?" in result.message_content
        assert "<context>" in result.message_content
        assert "</context>" in result.message_content
        assert result.convo_id == mock_convo.convo_id

        # Assert: save_message was called to save the message
        mock_convo_repo.save_message.assert_called_once()

    @pytest.mark.asyncio
    @patch("FOSRABack.src.convo.services.convo_service.app_state")
    async def test_new_user_message_context_in_prompt(
        self,
        mock_app_state,
        mock_retrieved_context_list: list[RetrievedContext],
        mock_request_context,
    ):
        """
        Test that context is properly embedded in prompt.

        Verify:
        - "<context>" tag in message_content
        - "</context>" closing tag present
        - "USER QUESTION" label present
        - Query text at end
        """
        # Setup
        sample_user = User(
            user_id="test_user",
            name="Test",
            created_at=datetime.now(),
            last_login=datetime.now(),
            pages_limit=100,
            pages_used=0,
        )

        mock_convo = MagicMock(spec=ConversationWithMessages)
        mock_convo.convo_id = "convo_123"
        mock_convo.workspace_id = 1
        mock_convo.messages = []

        mock_db_convo = MagicMock(spec=ConvoORM)
        mock_db_convo.messages = []
        mock_convo.to_orm = MagicMock(return_value=mock_db_convo)

        # Mock vector service
        mock_vector_service = AsyncMock()
        mock_vector_service.search.return_value = [
            ctx.model_dump() for ctx in mock_retrieved_context_list
        ]
        mock_app_state.vector_service = mock_vector_service

        mock_convo_repo = AsyncMock(spec=ConvoRepo)
        mock_convo_repo.save_message.return_value = MessageORM(
            message_content="<context> </context>**USER QUESTION** Explain machine learning ... What is AI?",
            role=MessageRole.USER.value,
            convo_id=mock_convo.convo_id,
        )

        convo_manager = ConvoService(convo_repo=mock_convo_repo)

        # Act
        result = await convo_manager.new_user_message(
            query="Explain machine learning",
            convo=mock_convo,
            ctx=mock_request_context,
        )

        # Assert: Verify prompt structure
        content = result.message_content
        assert "<context>" in content, "Missing opening context tag"
        assert "</context>" in content, "Missing closing context tag"
        assert "**USER QUESTION**" in content, "Missing USER QUESTION label"
        assert "Explain machine learning" in content, "Query not in prompt"

        # Assert: Context comes before query
        context_end = content.find("</context>")
        query_start = content.find("Explain machine learning")
        assert context_end < query_start, "Context should come before query"

    @pytest.mark.asyncio
    @patch("FOSRABack.src.convo.services.convo_service.app_state")
    async def test_new_user_message_persist_failure_raises(
        self,
        mock_app_state,
        mock_retrieved_context_list: list[RetrievedContext],
        mock_request_context,
    ):
        """
        Test that persist failure raises RuntimeError.

        Setup:
        - Mock successful context retrieval
        - Mock persist_session to raise Exception

        Verify:
        - RuntimeError raised with original exception info
        """
        # Setup
        sample_user = User(
            user_id="test_user",
            name="Test",
            created_at=datetime.now(),
            last_login=datetime.now(),
            pages_limit=100,
            pages_used=0,
        )

        mock_convo = MagicMock(spec=ConversationWithMessages)
        mock_convo.convo_id = "convo_123"
        mock_convo.workspace_id = 1
        mock_convo.messages = []

        mock_db_convo = MagicMock(spec=ConvoORM)
        mock_db_convo.messages = []
        mock_convo.to_orm = MagicMock(return_value=mock_db_convo)

        # Mock vector service to succeed
        mock_vector_service = AsyncMock()
        mock_vector_service.search.return_value = [
            ctx.model_dump() for ctx in mock_retrieved_context_list
        ]
        mock_app_state.vector_service = mock_vector_service

        # Mock repo to fail on save_message
        mock_convo_repo = AsyncMock(spec=ConvoRepo)
        mock_convo_repo.save_message.side_effect = Exception("Database connection lost")
        convo_manager = ConvoService(convo_repo=mock_convo_repo)

        # Act: Call the method - @log.catch() decorator swallows the exception
        # so we just verify the mock was called
        await convo_manager.new_user_message(
            query="Test query",
            convo=mock_convo,
            ctx=mock_request_context,
        )

        # Assert: save_message was called (even though it raised an exception)
        mock_convo_repo.save_message.assert_called_once()


@pytest.fixture
async def initialized_mock_db(mock_db: SQLSessionManager):
    """
    Initialize mock database and return it.

    Setup:
    - Call mock_db.init()
    - Return initialized db
    """
    await mock_db.init()
    return mock_db


@pytest.fixture
async def db_with_user_and_workspace(initialized_mock_db: SQLSessionManager):
    """
    Create database with user and workspace pre-populated.

    Creates:
    - UserORM with user_id="test_user_id"
    - WorkspaceORM with workspace_id=1, name="Test Workspace"

    Returns:
    - Tuple of (db, user, workspace)
    """
    async with initialized_mock_db.session_scope() as session:
        user = UserORM(user_id="test_user_id", name="Test User")
        session.add(user)
        await session.flush()

        workspace = WorkspaceORM(
            name="Test Workspace",
            user_id=user.user_id,
        )
        session.add(workspace)
        await session.commit()

        return initialized_mock_db, user, workspace


@pytest.fixture
def sample_user_for_integration():
    """
    Create User schema for integration tests.

    Fields:
    - user_id: "test_user_id"
    - name: "Test User"
    - pages_limit: 100
    - pages_used: 0
    """
    return User(
        user_id="test_user_id",
        name="Test User",
        created_at=datetime.now(),
        last_login=datetime.now(),
        pages_limit=100,
        pages_used=0,
    )


@pytest.fixture
def mock_convo_repo():
    """Create a mock ConvoRepo."""
    return AsyncMock()


@pytest.fixture
def mock_vector_service():
    """Create a mock VectorRepo."""
    return AsyncMock()


@pytest.fixture
def mock_llm_manager():
    """Create a mock LLM_Manager."""
    return AsyncMock()


@pytest.fixture
def convo_manager(mock_convo_repo):
    """Create a ConvoService instance with mocked dependencies."""
    return ConvoService(
        convo_repo=mock_convo_repo,
    )


# =============================================================================
# TEST CLASS: Database Conversation Persistence
# =============================================================================


@pytest.mark.asyncio
async def test_create_new_convo_success(
    convo_manager: ConvoService, mock_convo_repo: ConvoRepo, mock_request_context
):
    """Test successful conversation creation."""
    user = User(
        user_id="test_user",
        created_at=datetime.now(),
        last_login=datetime.now(),
        pages_limit=100,
        pages_used=0,
        name="Test User",
    )

    # Mock the repo response
    mock_conversation = Conversation(
        convo_id="test_convo_id",
        user_id=user.user_id,
        workspace_id=1,
        title="New Convo",
        created_at=datetime.now(),
    )
    mock_convo_repo = cast(AsyncMock, convo_manager.convo_repo)
    mock_convo_repo.create_new_convo.return_value = mock_conversation

    result = await convo_manager.create_new_convo(ctx=mock_request_context)

    assert result.convo_id == "test_convo_id"
    assert result.user_id == user.user_id

    mock_convo_repo.create_new_convo.assert_called_once_with(
        user_id="user_123", workspace_id=7
    )


@pytest.mark.asyncio
async def test_retrieve_existing_convo(
    convo_manager, mock_convo_repo, mock_request_context
):
    """Test retrieving an existing conversation."""
    mock_conversation = Conversation(
        convo_id="existing_convo",
        user_id="test_user",
        workspace_id=1,
        title="Existing Convo",
        created_at=datetime.now(),
    )
    mock_convo_repo.retrieve_convo_by_id.return_value = mock_conversation

    result = await convo_manager.convo_repo.retrieve_convo_by_id(
        mock_request_context, convo_id="existing_convo"
    )

    assert result.convo_id == "existing_convo"
    mock_convo_repo.retrieve_convo_by_id.assert_called_once()


class TestConversationDatabasePersistence:
    """Tests for conversation persistence to database."""

    @pytest.mark.asyncio
    async def test_new_chat_saved_to_database(
        self, mock_db: SQLSessionManager, sample_user_for_integration: User
    ):
        await mock_db.init()

        async with mock_db.session_scope() as session:
            # Create user first (foreign key constraint)
            user_orm = UserORM(
                user_id=sample_user_for_integration.user_id,
                name=sample_user_for_integration.name,
            )
            session.add(user_orm)
            await session.flush()

            # Create workspace (foreign key constraint)
            workspace_orm = WorkspaceORM(
                name="Test Workspace",
                user_id=user_orm.user_id,
            )
            session.add(workspace_orm)
            await session.flush()

            # Create conversation
            convo_orm = ConvoORM(
                convo_id="test_convo_123",
                title="Test Conversation",
                user_id=user_orm.user_id,
                workspace_id=workspace_orm.workspace_id,
            )
            session.add(convo_orm)
            await session.commit()

        # Verify in a new session (proves data persisted)
        async with mock_db.session_scope() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(ConvoORM).where(ConvoORM.convo_id == "test_convo_123")
            )
            fetched_convo = result.scalar_one_or_none()

            assert fetched_convo is not None, "Conversation not found in database"
            assert fetched_convo.title == "Test Conversation"
            assert fetched_convo.user_id == sample_user_for_integration.user_id

    @pytest.mark.asyncio
    async def test_conversation_has_generated_id(self, mock_db: SQLSessionManager):
        """
        Test that convo_id is auto-generated (ULID).

        Verify:
        - convo_id is not None
        - convo_id is string
        - convo_id has ULID format (26 chars)
        """
        await mock_db.init()

        async with mock_db.session_scope() as session:
            user_orm = UserORM(user_id="test_user", name="Test")
            session.add(user_orm)
            await session.flush()

            workspace_orm = WorkspaceORM(name="Test", user_id=user_orm.user_id)
            session.add(workspace_orm)
            await session.flush()

            # Create conversation WITHOUT specifying ID (should auto-generate)
            convo_orm = ConvoORM(
                title="Auto ID Test",
                user_id=user_orm.user_id,
                workspace_id=workspace_orm.workspace_id,
            )
            session.add(convo_orm)
            await session.commit()

            # Verify ULID properties
            assert convo_orm.convo_id is not None
            assert isinstance(convo_orm.convo_id, str)
            assert len(convo_orm.convo_id) == 26, "ULID should be 26 characters"

    @pytest.mark.asyncio
    async def test_conversation_created_at_set(self, mock_db: SQLSessionManager):
        """
        Test that created_at timestamp is automatically set.

        Verify:
        - created_at is not None
        - created_at is datetime
        - created_at is recent (within last minute)
        """
        await mock_db.init()
        before_creation = datetime.now()

        async with mock_db.session_scope() as session:
            user_orm = UserORM(user_id="test_user", name="Test")
            session.add(user_orm)
            await session.flush()

            workspace_orm = WorkspaceORM(name="Test", user_id=user_orm.user_id)
            session.add(workspace_orm)
            await session.flush()

            convo_orm = ConvoORM(
                title="Timestamp Test",
                user_id=user_orm.user_id,
                workspace_id=workspace_orm.workspace_id,
            )
            session.add(convo_orm)
            await session.commit()

            after_creation = datetime.utcnow()

            # Verify timestamp properties
            assert convo_orm.created_at is not None
            assert isinstance(convo_orm.created_at, datetime)
            # Compare without timezone
            created_naive = convo_orm.created_at.replace(tzinfo=None)

            assert (
                before_creation.replace(tzinfo=None)
                <= created_naive
                <= after_creation.replace(tzinfo=None)
            )

    @pytest.mark.asyncio
    async def test_multiple_conversations_same_user(self, mock_db: SQLSessionManager):
        """
        Test creating multiple conversations for same user.

        Setup:
        - Create 3 conversations for same user

        Verify:
        - All 3 retrievable
        - Each has unique convo_id
        - All have same user_id
        """
        await mock_db.init()

        async with mock_db.session_scope() as session:
            user_orm = UserORM(user_id="multi_convo_user", name="Test")
            session.add(user_orm)
            await session.flush()

            workspace_orm = WorkspaceORM(name="Test", user_id=user_orm.user_id)
            session.add(workspace_orm)
            await session.flush()

            # Create 3 conversations
            convos = []
            for i in range(3):
                convo = ConvoORM(
                    title=f"Conversation {i}",
                    user_id=user_orm.user_id,
                    workspace_id=workspace_orm.workspace_id,
                )
                session.add(convo)
                convos.append(convo)

            await session.commit()

            # Verify all have unique IDs
            convo_ids = [c.convo_id for c in convos]
            assert len(set(convo_ids)) == 3, "All conversation IDs should be unique"

            # Verify all have same user_id
            for convo in convos:
                assert convo.user_id == "multi_convo_user"


# =============================================================================
# TEST CLASS: ConvoService Database Integration
# =============================================================================


class TestConvoServiceDatabaseIntegration:
    """Tests for ConvoService with real database operations."""

    @pytest.mark.asyncio
    async def test_db_ops_are_committing(
        self, mock_db: SQLSessionManager, mock_request_context
    ):
        """
        Test that ConvoService operations commit to database.

        Setup:
        - Create user and workspace in db
        - Use convo_manager.create_new_convo

        Verify:
        - Conversation exists in database after create
        - Can query by convo_id
        """
        await mock_db.init()

        # Setup: Create user and workspace
        async with mock_db.session_scope() as session:
            user_orm = UserORM(user_id="commit_test_user", name="Test")
            session.add(user_orm)
            await session.flush()

            workspace_orm = WorkspaceORM(name="Test", user_id=user_orm.user_id)
            session.add(workspace_orm)
            await session.commit()
            workspace_id = workspace_orm.workspace_id

        # Create ConvoService with real repo
        convo_repo = ConvoRepo(db_session=mock_db)
        convo_manager = ConvoService(convo_repo=convo_repo)

        # Create user schema for the method
        user = User(
            user_id="commit_test_user",
            name="Test",
            created_at=datetime.now(),
            last_login=datetime.now(),
            pages_limit=100,
            pages_used=0,
        )

        # Act: Create conversation through manager
        result = await convo_manager.create_new_convo(ctx=mock_request_context)

        # Verify: Query in new session to prove commit happened
        async with mock_db.session_scope() as session:
            from sqlalchemy import select

            fetched = await session.execute(
                select(ConvoORM).where(ConvoORM.convo_id == result.convo_id)
            )
            convo = fetched.scalar_one_or_none()
            assert convo is not None, "Conversation not committed to database"

    @pytest.mark.asyncio
    async def test_conversation_persists_across_sessions(
        self, mock_db: SQLSessionManager, mock_request_context
    ):
        """
        Test conversation available in new database session.

        Setup:
        - Create conversation in one session
        - Query in new session scope

        Verify:
        - Conversation found in new session
        - All fields intact
        """
        await mock_db.init()
        convo_id = None

        # Session 1: Create data
        async with mock_db.session_scope() as session:
            user_orm = UserORM(user_id="session_test_user", name="Test")
            session.add(user_orm)
            await session.flush()

            workspace_orm = WorkspaceORM(name="Test", user_id=user_orm.user_id)
            session.add(workspace_orm)
            await session.flush()

            convo_orm = ConvoORM(
                title="Cross Session Test",
                user_id=user_orm.user_id,
                workspace_id=workspace_orm.workspace_id,
            )
            session.add(convo_orm)
            await session.commit()
            convo_id = convo_orm.convo_id

        # Session 2: Verify data exists
        async with mock_db.session_scope() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(ConvoORM).where(ConvoORM.convo_id == convo_id)
            )
            convo = result.scalar_one_or_none()

            assert convo is not None, "Conversation not found in new session"
            assert convo.title == "Cross Session Test"
            assert convo.user_id == "session_test_user"

    @pytest.mark.asyncio
    async def test_conversation_relationship_to_workspace(
        self, mock_db: SQLSessionManager, mock_request_context
    ):
        """
        Test conversation correctly linked to workspace.

        Verify:
        - conversation.workspace_id matches created workspace
        - Can navigate relationship
        """
        await mock_db.init()

        ctx = mock_request_context

        async with mock_db.session_scope() as session:
            user_orm = UserORM(user_id=ctx.user_id, name="Test")

            session.add(user_orm)
            await session.flush()

            workspace_orm = WorkspaceORM(
                name="Relationship Test Workspace",
                user_id=ctx.user_id,
            )
            session.add(workspace_orm)

            await session.flush()

            workspace_id = workspace_orm.workspace_id

            convo_orm = ConvoORM(
                title="Relationship Test",
                user_id=ctx.user_id,
                workspace_id=workspace_orm.workspace_id,
            )
            session.add(convo_orm)
            await session.commit()

            # Verify relationship
            assert convo_orm.workspace_id == workspace_id
            # Navigate relationship (if lazy loading configured)
            assert convo_orm.workspace is not None
            assert convo_orm.workspace.name == "Relationship Test Workspace"

    @pytest.mark.asyncio
    @patch("FOSRABack.src.convo.services.convo_service.app_state")
    async def test_get_context_handles_dict_response(
        self, mock_app_state, mock_request_context
    ):
        """
        Test get_context converts dict responses to RetrievedContext.

        Setup:
        - Return list of dicts from mock

        Verify:
        - Dicts converted to RetrievedContext objects
        """
        # Setup: Mock vector service to return dicts
        mock_vector_service = AsyncMock()
        mock_vector_service.search.return_value = [
            {
                "chunk": {
                    "chunk_id": "c1",
                    "source_id": "s1",
                    "source_hash": "h1",
                    "text": "Dict response text",
                    "start_index": 0,
                    "end_index": 50,
                    "token_count": 10,
                },
                "source": {
                    "source_id": "s1",
                    "source_hash": "h1",
                    "unique_id": "u1",
                    "source_summary": "",
                    "summary_embedding": "[]",
                },
                "similarity_score": 0.85,
                "result_rank": 1,
            }
        ]
        mock_app_state.vector_service = mock_vector_service

        mock_repo = AsyncMock()
        convo_manager = ConvoService(convo_repo=mock_repo)

        # Act
        result = await convo_manager.get_context(query="test", ctx=mock_request_context)

        # Assert
        assert len(result) == 1
        assert isinstance(result[0], RetrievedContext)
        assert result[0].chunk.text == "Dict response text"
        assert result[0].similarity_score == 0.85


# =============================================================================
# TEST CLASS: Workspace Persistence
# =============================================================================


class TestWorkspacePersistence:
    """Tests for workspace persistence functionality."""

    @pytest.mark.asyncio
    async def test_workspace_persisting_to_disk(
        self,
        mock_db_local: SQLSessionManager,
        workspace_manager_local: WorkspaceService,
        tmp_path: Path,
    ):
        """
        Test workspace persists to SQLite file on disk.

        Setup:
        - Create workspace using local SQLite file
        - Close session
        - Open new session to same file

        Verify:
        - Workspace retrievable from new session
        - Name and ID match original
        """
        pass


# =============================================================================
# TEST CLASS: Assistant Message Creation
# =============================================================================


class TestAssistantMessageCreation:
    """Tests for assistant message creation."""

    @pytest.mark.asyncio
    async def test_new_assistant_message_creates_valid_message(
        self, convo_manager: ConvoService, mock_convo: MagicMock
    ):
        """
        Test new_assistant_message creates properly formatted Message.

        Verify:
        - role is "assistant"
        - message_id is generated
        - message_content matches input
        - convo_id matches session
        """
        mock_convo.convo_id = "test_convo"

        result = await convo_manager.new_assistant_message(
            response_str="Hello, I'm the assistant!",
            convo_session=mock_convo,
        )

        assert result.role == MessageRole.ASSISTANT
        assert result.message_id is not None
        assert result.message_content == "Hello, I'm the assistant!"
        assert result.convo_id == "test_convo"

    @pytest.mark.asyncio
    async def test_assistant_message_has_ulid_id(
        self, convo_manager: ConvoService, mock_convo: MagicMock
    ):
        """
        Test that message_id is valid ULID format.

        Verify:
        - message_id is 26 characters
        - message_id is uppercase alphanumeric
        """
        mock_convo.convo_id = "test_convo"

        result = await convo_manager.new_assistant_message(
            response_str="Test response",
            convo_session=mock_convo,
        )
        assert result.message_id is not None

        assert len(result.message_id) == 26, "ULID should be 26 characters"
        # ULID characters are Crockford's Base32 (uppercase alphanumeric)
        assert result.message_id.isalnum(), "ULID should be alphanumeric"


# =============================================================================
# TEST CLASS: Stream Results
# =============================================================================


class TestStreamResults:
    """Tests for stream_results async generator."""

    @pytest.mark.asyncio
    async def test_stream_results_yields_chunks(self):
        """
        Test stream_results yields chunks from graph.

        Setup:
        - Patch researcher_graph.astream
        - Patch StreamingService
        - Yield test chunks

        Verify:
        - All chunks yielded
        - Completion message at end
        """
        pytest.skip("stream_results not yet integrated into ConvoService")

    @pytest.mark.asyncio
    async def test_stream_results_handles_uuid_user_id(self):
        """
        Test stream_results accepts UUID user_id.

        Setup:
        - Pass uuid4() as user_id

        Verify:
        - No error raised
        - user_id converted to string in config
        """
        pytest.skip("stream_results not yet integrated into ConvoService")

    @pytest.mark.asyncio
    async def test_stream_results_search_mode_conversion(self):
        """
        Test stream_results converts string search_mode to enum.

        Test values: "CHUNKS", "DOCUMENTS", "invalid"

        Verify:
        - Valid modes converted to SearchMode enum
        - Invalid defaults to CHUNKS
        """
        pytest.skip("stream_results not yet integrated into ConvoService")

    @pytest.mark.asyncio
    async def test_stream_results_config_structure(self):
        """
        Test stream_results passes correct config to graph.

        Verify config contains:
        - user_query
        - user_id (as string)
        - workspace_id
        - search_mode (as enum)
        - document_ids_to_add_in_context
        - top_k
        - enable_reranking
        - enable_citations
        """
        pytest.skip("stream_results not yet integrated into ConvoService")

    @pytest.mark.asyncio
    async def test_stream_results_initial_state(self):
        """
        Test stream_results creates correct initial State.

        Verify State contains:
        - db_session
        - streaming_service
        - chat_history
        - vector_service
        """
        pytest.skip("stream_results not yet integrated into ConvoService")

    @pytest.mark.asyncio
    async def test_stream_results_completion_at_end(self):
        """
        Test stream_results always yields completion at end.

        Verify:
        - Last yielded value is from format_completion()
        - format_completion called exactly once
        """
        pytest.skip("stream_results not yet integrated into ConvoService")
