# Comprehensive test fixtures and mocks
import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime
from typing import Dict, List, Any


class MockUser:
    """Mock user for testing"""

    def __init__(self, id: str = "test-user-id", email: str = "test@example.com"):
        self.user_id: str = id
        self.name = "Test User"
        self.created_at: datetime = datetime.now()
        self.last_login: datetime = datetime.now()
        self.enabled: bool = True
        self.pages_limit: int = 500
        self.pages_used: int = 0
        self.workspaces = []
        self.llm_config = []


class MockWorkSpace:
    """Mock search space for testing"""

    def __init__(
        self,
        id: int = 1,
        name: str = "Test Search Space",
        user_id: str = "test-user-id",
    ):
        self.id = id
        self.name = name
        self.description = "Test search space description"
        self.user_id = user_id
        self.created_at = datetime.now()
        self.user = None  # Avoid circular dependency


class MockSource:
    """Mock document for testing"""

    def __init__(
        self, id: int = 1, title: str = "Test Document", search_space_id: int = 1
    ):
        self.id = id
        self.title = title
        self.document_type = "FILE"
        self.content = "Test document content"
        self.content_hash = f"hash_{id}"
        self.unique_identifier_hash = f"unique_hash_{id}"
        self.search_space_id = search_space_id
        self.created_at = datetime.now()
        self.chunks = []


class MockChunk:
    """Mock chunk for testing"""

    def __init__(
        self, id: int = 1, content: str = "Test chunk content", document_id: int = 1
    ):
        self.id = id
        self.content = content
        self.document_id = document_id
        self.created_at = datetime.now()


class MockConvo:
    """Mock chat for testing"""

    def __init__(self, id: int = 1, title: str = "Test Chat", search_space_id: int = 1):
        self.convo_id = id
        self.type = "QNA"
        self.title = title
        self.messages = [{"role": "user", "content": "Hello"}]
        self.search_space_id = search_space_id
        self.state_version = 1
        self.created_at = datetime.now()


class MockLLMConfig:
    """Mock LLM config for testing"""

    def __init__(
        self, id: int = 1, name: str = "Test LLM Config", search_space_id: int = 1
    ):
        self.id = id
        self.name = name
        self.provider = "OPENAI"
        self.model_name = "gpt-3.5-turbo"
        self.api_key = "test-api-key"
        self.language = "English"
        self.search_space_id = search_space_id
        self.created_at = datetime.now()


class MockLLMResponse:
    """Mock LLM response for testing"""

    def __init__(self, content: str = "Test LLM response"):
        self.content = content


class MockRetrievalJob:
    """Mock retriever response for testing"""

    def __init__(self, content: str = "Retrieved content", score: float = 0.9):
        self.content = content
        self.score = score
        self.metadata = {}


class MockRerankerResponse:
    """Mock reranker response for testing"""

    def __init__(self, content: str = "Reranked content", score: float = 0.95):
        self.content = content
        self.score = score


class MockAgentResponse:
    """Mock agent response for testing"""

    def __init__(
        self,
        answer: str = "Test answer",
        documents: list[dict[Any, Any]] | None = None,
        sources: list[dict[Any, Any]] | None = None,
    ):
        self.answer = answer
        self.documents = documents or [{"content": "Test document"}]
        self.sources = sources or [{"title": "Test source"}]


class MockStreamingChunk:
    """Mock streaming chunk for testing"""

    def __init__(self, data: str = "test data"):
        self.data = data.encode() if isinstance(data, str) else data


class TestDataFactory:
    """Factory for creating test data"""

    @staticmethod
    def create_user_data(**overrides) -> Dict[str, Any]:
        """Create user data with optional overrides"""
        defaults = {
            "id": "test-user-id",
            "email": "test@example.com",
            "is_active": True,
            "is_verified": True,
            "pages_limit": 500,
            "pages_used": 0,
        }
        defaults.update(overrides)
        return defaults

    @staticmethod
    def create_search_space_data(**overrides) -> Dict[str, Any]:
        """Create search space data with optional overrides"""
        defaults = {
            "id": 1,
            "name": "Test Search Space",
            "description": "Test search space description",
            "user_id": "test-user-id",
        }
        defaults.update(overrides)
        return defaults

    @staticmethod
    def create_document_data(**overrides) -> Dict[str, Any]:
        """Create document data with optional overrides"""
        defaults = {
            "id": 1,
            "title": "Test Document",
            "document_type": "FILE",
            "content": "Test document content",
            "content_hash": "test_hash_123",
            "search_space_id": 1,
        }
        defaults.update(overrides)
        return defaults

    @staticmethod
    def create_chat_data(**overrides) -> Dict[str, Any]:
        """Create chat data with optional overrides"""
        defaults = {
            "id": 1,
            "type": "QNA",
            "title": "Test Chat",
            "messages": [{"role": "user", "content": "Hello"}],
            "search_space_id": 1,
        }
        defaults.update(overrides)
        return defaults


class MockServiceFactory:
    """Factory for creating mock services"""

    @staticmethod
    def create_mock_llm_service(response: str | None = None) -> Mock:
        """Create mock LLM service"""
        service = Mock()
        service.completion = Mock()
        service.completion.return_value = Mock()
        service.completion.return_value.choices = [Mock()]
        service.completion.return_value.choices[0].message.content = (
            response or "Test LLM response"
        )
        return service

    @staticmethod
    def create_mock_retriever_service(
        documents: list[dict[Any, Any]] | None = None,
    ) -> Mock:
        """Create mock retriever service"""
        service = Mock()
        service.aretrieve = AsyncMock(
            return_value=documents or [mock_retriever_service()().__dict__]
        )
        return service

    @staticmethod
    def create_mock_reranker_service(documents: List[Dict] | None = None) -> Mock:
        """Create mock reranker service"""
        service = Mock()
        service.rerank_documents = Mock(
            return_value=documents or [MockRerankerResponse().__dict__]
        )
        return service

    @staticmethod
    def create_mock_streaming_service() -> Mock:
        """Create mock streaming service"""
        service = Mock()
        service.format_answer_delta = Mock(return_value="data: test answer\n\n")
        service.format_sources_delta = Mock(return_value="data: test sources\n\n")
        service.format_terminal_info_delta = Mock(return_value="data: test info\n\n")
        service.format_error = Mock(return_value="data: test error\n\n")
        service.format_completion = Mock(return_value="data: completed\n\n")
        return service

    @staticmethod
    def create_mock_qna_agent(response: Dict | None = None) -> Mock:
        """Create mock QnA agent"""
        agent = Mock()
        agent.answer_question = AsyncMock(
            return_value=response or MockAgentResponse().__dict__
        )
        return agent

    @staticmethod
    def create_mock_researcher_agent(response: Dict | None = None) -> Mock:
        """Create mock researcher agent"""
        agent = Mock()
        agent.research = AsyncMock(
            return_value=response or MockAgentResponse().__dict__
        )
        return agent


class MockAsyncSession:
    """Mock async session for testing"""

    def __init__(self):
        self._objects = {}
        self._committed = False
        self._rolled_back = False

    def add(self, obj):
        """Add object to session"""
        if hasattr(obj, "id"):
            self._objects[obj.id] = obj

    async def commit(self):
        """Commit session"""
        self._committed = True

    async def rollback(self):
        """Rollback session"""
        self._rolled_back = True

    async def refresh(self, obj):
        """Refresh object"""
        pass

    async def delete(self, obj):
        """Delete object"""
        if hasattr(obj, "id") and obj.id in self._objects:
            del self._objects[obj.id]

    async def execute(self, query):
        """Execute query (mocked)"""
        return Mock()

    def scalars(self):
        """Get scalars result (mocked)"""
        return Mock()

    def first(self):
        """Get first result (mocked)"""
        return None

    def all(self):
        """Get all results (mocked)"""
        return list(self._objects.values())


@pytest.fixture
def mock_user():
    """Create mock user"""
    return MockUser()


@pytest.fixture
def mock_workspace():
    """Create mock search space"""
    return MockWorkSpace()


@pytest.fixture
def mock_Source():
    """Create mock document"""
    return MockSource()


@pytest.fixture
def mock_convo():
    """Create mock chat"""
    return MockConvo()


@pytest.fixture
def mock_user_llm_config():
    """Create mock chat"""
    return MockLLMConfig()


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service"""
    return MockServiceFactory.create_mock_llm_service()


@pytest.fixture
def mock_retriever_service():
    """Create mock retriever service"""
    return MockServiceFactory.create_mock_retriever_service()


@pytest.fixture
def mock_reranker_service():
    """Create mock reranker service"""
    return MockServiceFactory.create_mock_reranker_service()


@pytest.fixture
def mock_streaming_service():
    """Create mock streaming service"""
    return MockServiceFactory.create_mock_streaming_service()


@pytest.fixture
def mock_qna_agent():
    """Create mock QnA agent"""
    return MockServiceFactory.create_mock_qna_agent()


@pytest.fixture
def mock_researcher_agent():
    """Create mock researcher agent"""
    return MockServiceFactory.create_mock_researcher_agent()


@pytest.fixture
def mock_async_session():
    """Create mock async session"""
    return MockAsyncSession()


@pytest.fixture
def test_data_factory():
    """Provide test data factory"""
    return TestDataFactory()


@pytest.fixture
def mock_service_factory():
    """Provide mock service factory"""
    return MockServiceFactory()
