from typing import Any
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from backend.src.domain.exceptions import (
    # Base
    FOSRAError,
    # Infrastructure (500/503)
    InfrastructureError,
    LLMRateLimitError,
    # Security (401/403)
    SecurityError,
    TenantContextError,
    ProcessingError,
    ParsingError,
    UnsupportedFileTypeError,
    ChunkingError,
    ContextWindowError,
    # Storage (404/500)
    StorageError,
    FileStorageError,
    UserExistenceError,
    WorkspaceExistenceError,
    CollectionNotFoundError,
    EmptyRetrievalError,
    # Conversation (400/500)
    ConvoError,
    SessionNotFoundError,
    StateTransitionError,
    ConvoValidationError,
    LLMError,
    LLMValidationError,
    LLMConfigurationError,
    # Embedder (500/503)
    EmbedderError,
    EmbedderNotFoundError,
    APIKeyMissingError,
    ModelNotInitializedError,
    # Parser (422/500)
    ParserError,
    ParserNotFoundError,
    UnsupportedDocumentTypeError,
    ParserDependencyError,
    InvalidDocumentError,
    # File Storage Backend (404/500)
    StorageBackendNotFoundError,
    FileNotFoundError,
    InvalidPathError,
    UnsupportedBackendError,
    StorageCredentialsError,
    RerankerError,
    RerankerNotFoundError,
    RerankerAPIConfigError,
    InvalidDocumentListError,
    RerankerDependencyError,
    # Query & Search (400/404)
    QueryError,
    QueryExpansionError,
    QueryReformulationError,
    RetrievalError,
    ChunkRetrievalError,
    SourceRetrievalError,
    RAGError,
)


def create_error_response(
    exc: FOSRAError,
    status_code: int,
    request: Request,
) -> JSONResponse:
    """Create standardized error response."""

    error_detail: dict[str, str | dict[str, Any]] = {
        "error": exc.__class__.__name__,
        "message": exc.message,
        "path": str(request.url.path),
    }

    # Add optional fields if present
    if exc.remediation:
        error_detail["remediation"] = exc.remediation

    if exc.docs_url:
        error_detail["docs_url"] = exc.docs_url

    if exc.detail:
        error_detail["detail"] = exc.detail

    # Log error with appropriate level
    if status_code >= 500:
        logger.error(f"{exc.__class__.__name__}: {exc.message}")
    else:
        logger.warning(f"{exc.__class__.__name__}: {exc.message}")

    return JSONResponse(
        status_code=status_code,
        content=error_detail,
    )


# =============================================================================
# Exception Handlers by Category
# =============================================================================


async def infrastructure_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle infrastructure and resource errors (500/503)."""
    if not isinstance(exc, InfrastructureError):
        raise exc
    # Rate limits return 429, others return 503
    if isinstance(exc, LLMRateLimitError):
        return create_error_response(exc, status.HTTP_429_TOO_MANY_REQUESTS, request)
    return create_error_response(exc, status.HTTP_503_SERVICE_UNAVAILABLE, request)


async def security_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle security and access control errors (401/403)."""
    if not isinstance(exc, SecurityError):
        raise exc
    if isinstance(exc, TenantContextError):
        return create_error_response(exc, status.HTTP_400_BAD_REQUEST, request)
    return create_error_response(exc, status.HTTP_403_FORBIDDEN, request)


async def processing_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle processing and ingestion errors (422/400)."""
    if not isinstance(exc, ProcessingError):
        raise exc
    # Unsupported file types and parsing errors are unprocessable (422)
    if isinstance(exc, (UnsupportedFileTypeError, ParsingError, ChunkingError)):
        return create_error_response(exc, status.HTTP_422_UNPROCESSABLE_ENTITY, request)
    # Context window errors are bad requests (400)
    if isinstance(exc, ContextWindowError):
        return create_error_response(exc, status.HTTP_400_BAD_REQUEST, request)
    # Default to internal server error (500)
    return create_error_response(exc, status.HTTP_500_INTERNAL_SERVER_ERROR, request)


async def storage_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle storage and retrieval errors (404/500)."""
    if not isinstance(exc, StorageError):
        raise exc
    # Existence errors return 404
    existence_errors = (
        UserExistenceError,
        WorkspaceExistenceError,
        FileNotFoundError,
        CollectionNotFoundError,
        SessionNotFoundError,
    )
    if isinstance(exc, existence_errors):
        return create_error_response(exc, status.HTTP_404_NOT_FOUND, request)

    # Empty retrieval can return 404 or trigger fallback (depends on context)
    if isinstance(exc, EmptyRetrievalError):
        return create_error_response(exc, status.HTTP_404_NOT_FOUND, request)

    # All other storage errors return 500
    return create_error_response(exc, status.HTTP_500_INTERNAL_SERVER_ERROR, request)


async def convo_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle conversation and streaming errors (400/500)."""
    if not isinstance(exc, ConvoError):
        raise exc
    # Validation errors return 400
    if isinstance(exc, (ConvoValidationError, StateTransitionError)):
        return create_error_response(exc, status.HTTP_400_BAD_REQUEST, request)

    # Session not found returns 404
    if isinstance(exc, SessionNotFoundError):
        return create_error_response(exc, status.HTTP_404_NOT_FOUND, request)

    # State and streaming errors return 500
    return create_error_response(exc, status.HTTP_500_INTERNAL_SERVER_ERROR, request)


async def llm_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle LLM operation errors (500/503)."""
    if not isinstance(exc, LLMError):
        raise exc
    # Validation errors return 400
    if isinstance(exc, LLMValidationError):
        return create_error_response(exc, status.HTTP_400_BAD_REQUEST, request)

    # Configuration errors return 500
    if isinstance(exc, LLMConfigurationError):
        return create_error_response(
            exc, status.HTTP_500_INTERNAL_SERVER_ERROR, request
        )

    # Provider errors return 503
    return create_error_response(exc, status.HTTP_503_SERVICE_UNAVAILABLE, request)


async def embedder_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle embedder operation errors (500/503)."""
    if not isinstance(exc, EmbedderError):
        raise exc
    # Missing API key returns 401
    if isinstance(exc, APIKeyMissingError):
        return create_error_response(exc, status.HTTP_401_UNAUTHORIZED, request)

    # Not found errors return 404
    if isinstance(exc, (EmbedderNotFoundError, ModelNotInitializedError)):
        return create_error_response(exc, status.HTTP_404_NOT_FOUND, request)

    # Initialization and operation errors return 503
    return create_error_response(exc, status.HTTP_503_SERVICE_UNAVAILABLE, request)


async def parser_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle parser operation errors (422/500)."""
    if not isinstance(exc, ParserError):
        raise exc
    # Unsupported document types and invalid documents return 422
    if isinstance(exc, (UnsupportedDocumentTypeError, InvalidDocumentError)):
        return create_error_response(exc, status.HTTP_422_UNPROCESSABLE_ENTITY, request)

    # Not found errors return 404
    if isinstance(exc, ParserNotFoundError):
        return create_error_response(exc, status.HTTP_404_NOT_FOUND, request)

    # Dependency errors return 503
    if isinstance(exc, ParserDependencyError):
        return create_error_response(exc, status.HTTP_503_SERVICE_UNAVAILABLE, request)

    # Parsing operation errors and timeouts return 500
    return create_error_response(exc, status.HTTP_500_INTERNAL_SERVER_ERROR, request)


async def file_storage_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle file storage backend errors (404/500)."""
    if not isinstance(exc, FileStorageError):
        raise exc
    # File not found and invalid path return 404
    if isinstance(exc, (FileNotFoundError, InvalidPathError)):
        return create_error_response(exc, status.HTTP_404_NOT_FOUND, request)

    # Backend not found and unsupported backend return 400
    if isinstance(exc, (StorageBackendNotFoundError, UnsupportedBackendError)):
        return create_error_response(exc, status.HTTP_400_BAD_REQUEST, request)

    # Credentials errors return 401
    if isinstance(exc, StorageCredentialsError):
        return create_error_response(exc, status.HTTP_401_UNAUTHORIZED, request)

    # Connection and other storage errors return 503
    return create_error_response(exc, status.HTTP_503_SERVICE_UNAVAILABLE, request)


async def reranker_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle reranker operation errors (500/503)."""
    if not isinstance(exc, RerankerError):
        raise exc
    # Not found errors return 404
    if isinstance(exc, RerankerNotFoundError):
        return create_error_response(exc, status.HTTP_404_NOT_FOUND, request)

    # Configuration errors return 400
    if isinstance(exc, (RerankerAPIConfigError, InvalidDocumentListError)):
        return create_error_response(exc, status.HTTP_400_BAD_REQUEST, request)

    # Dependency errors return 503
    if isinstance(exc, RerankerDependencyError):
        return create_error_response(exc, status.HTTP_503_SERVICE_UNAVAILABLE, request)

    # Operation errors and timeouts return 500
    return create_error_response(exc, status.HTTP_500_INTERNAL_SERVER_ERROR, request)


async def query_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle query and search errors (400/404)."""
    if not isinstance(exc, QueryError):
        raise exc
    # Query expansion and reformulation errors return 400
    if isinstance(exc, (QueryExpansionError, QueryReformulationError)):
        return create_error_response(exc, status.HTTP_400_BAD_REQUEST, request)

    # Default to 400 for query errors
    return create_error_response(exc, status.HTTP_400_BAD_REQUEST, request)


async def retrieval_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle retrieval operation errors (404/500)."""
    if not isinstance(exc, RetrievalError):
        raise exc
    # Empty results and not found errors return 404
    if isinstance(exc, (ChunkRetrievalError, SourceRetrievalError)):
        return create_error_response(exc, status.HTTP_404_NOT_FOUND, request)

    # Vector search and other retrieval errors return 500
    return create_error_response(exc, status.HTTP_500_INTERNAL_SERVER_ERROR, request)


async def rag_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle RAG-specific errors (500/503)."""
    if not isinstance(exc, RAGError):
        raise exc
    return create_error_response(exc, status.HTTP_500_INTERNAL_SERVER_ERROR, request)


async def base_fosra_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for any FOSRAError not caught by specific handlers."""
    if not isinstance(exc, FOSRAError):
        raise exc
    logger.error(f"Unhandled FOSRAError: {exc.__class__.__name__}: {exc.message}")
    return create_error_response(exc, status.HTTP_500_INTERNAL_SERVER_ERROR, request)


async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected exceptions."""
    logger.exception(f"Unexpected error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "path": str(request.url.path),
        },
    )


# =============================================================================
# Registration
# =============================================================================


def register_exception_handlers(app: FastAPI) -> None:
    # Infrastructure (500/503)
    app.add_exception_handler(InfrastructureError, infrastructure_error_handler)

    # Security (401/403)
    app.add_exception_handler(SecurityError, security_error_handler)

    # Processing (422/400)
    app.add_exception_handler(ProcessingError, processing_error_handler)

    # Storage (404/500)
    app.add_exception_handler(StorageError, storage_error_handler)

    # Conversation (400/500)
    app.add_exception_handler(ConvoError, convo_error_handler)

    # LLM Operations (500/503)
    app.add_exception_handler(LLMError, llm_error_handler)

    # Embedder (500/503)
    app.add_exception_handler(EmbedderError, embedder_error_handler)

    # Parser (422/500)
    app.add_exception_handler(ParserError, parser_error_handler)

    # File Storage (404/500)
    app.add_exception_handler(FileStorageError, file_storage_error_handler)

    # Reranker (500/503)
    app.add_exception_handler(RerankerError, reranker_error_handler)

    # Query & Search (400/404)
    app.add_exception_handler(QueryError, query_error_handler)

    app.add_exception_handler(RetrievalError, retrieval_error_handler)

    # RAG (500/503)
    app.add_exception_handler(RAGError, rag_error_handler)

    # Base FOSRA error (fallback for any FOSRAError)
    app.add_exception_handler(FOSRAError, base_fosra_error_handler)

    # Generic exception handler (catch-all)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Exception handlers registered successfully")
