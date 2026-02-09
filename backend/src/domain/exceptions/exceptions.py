from __future__ import annotations

from typing import Any


# =============================================================================
# Base Exception
# =============================================================================


class FOSRAError(Exception):
    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        docs_url: str | None = None,
        detail: dict | None = None,
    ):
        """Initialize FOSRA error."""

        super().__init__(message)
        self.message = message
        self.remediation = remediation
        self.docs_url = docs_url
        self.detail = detail or {}


# =============================================================================
# 1. Infrastructure & Resource Exceptions (500/503)
# =============================================================================


class InfrastructureError(FOSRAError):
    """Base for external service failures (DB, Vector Store, LLM, APIs)."""

    def __init__(
        self,
        service_name: str,
        reason: str,
        *,
        operation: str | None = None,
        remediation: str | None = None,
    ):
        self.service_name = service_name
        self.operation = operation

        message = f"{service_name} infrastructure error: {reason}"
        if operation:
            message += f" (operation: {operation})"

        if not remediation:
            remediation = f"Check {service_name} service availability and configuration"

        super().__init__(message=message, remediation=remediation)


class InitializationError(InfrastructureError):
    """Raised when a service or component fails to initialize."""

    def __init__(
        self,
        component_name: str,
        reason: str,
        *,
        component_type: str | None = None,
        remediation: str | None = None,
    ):
        self.component_name = component_name
        self.component_type = component_type

        message = f"Failed to initialize {component_name}"
        if component_type:
            message += f" ({component_type})"
        message += f": {reason}"

        if not remediation:
            remediation = f"Check {component_name} dependencies and configuration"

        super().__init__(
            service_name=component_name,
            reason=reason,
            operation="initialization",
            remediation=remediation,
        )


class ServiceUnavailableError(InfrastructureError):
    """Raised when a required service is unreachable or down."""

    def __init__(
        self,
        service_name: str,
        reason: str,
        *,
        endpoint: str | None = None,
        retry_after: int | None = None,
        remediation: str | None = None,
    ):
        self.endpoint = endpoint
        self.retry_after = retry_after

        message = f"Service unavailable: {service_name} - {reason}"
        if endpoint:
            message += f" (endpoint: {endpoint})"
        if retry_after:
            message += f" (retry after: {retry_after}s)"

        if not remediation:
            remediation = (
                f"Check {service_name} service status and network connectivity"
            )

        super().__init__(
            service_name=service_name,
            reason=reason,
            operation="connection",
            remediation=remediation,
        )


class LLMProviderError(InfrastructureError):
    """Raised when LLM provider returns errors (OpenAI, Anthropic, etc.)."""

    def __init__(
        self,
        provider: str,
        model: str,
        reason: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        remediation: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.error_type = error_type

        message = f"LLM provider error ({provider}/{model}): {reason}"
        if status_code:
            message += f" (status: {status_code})"
        if error_type:
            message += f" (type: {error_type})"

        if not remediation:
            if status_code == 401:
                remediation = f"Check {provider} API key configuration"
            elif status_code == 429:
                remediation = "Rate limit exceeded, wait before retrying"
            else:
                remediation = f"Check {provider} API status and model availability"

        super().__init__(
            service_name=f"{provider}/{model}",
            reason=reason,
            operation="llm_request",
            remediation=remediation,
        )


class LLMRateLimitError(InfrastructureError):
    """Raised when LLM provider rate limits are hit (429 errors)."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        retry_after: int | None = None,
        limit_type: str | None = None,
        remediation: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.retry_after = retry_after
        self.limit_type = limit_type

        message = f"Rate limit exceeded for {provider}/{model}"
        if limit_type:
            message += f" ({limit_type} limit)"
        if retry_after:
            message += f" (retry after: {retry_after}s)"

        if not remediation:
            remediation = (
                f"Wait {retry_after or 60}s before retrying, "
                "or upgrade your API plan for higher limits"
            )

        super().__init__(
            service_name=f"{provider}/{model}",
            reason="Rate limit exceeded",
            operation="llm_request",
            remediation=remediation,
        )


# =============================================================================
# 2. Security & Access Control Exceptions (401/403)
# =============================================================================


class SecurityError(FOSRAError):
    """Base for authentication and authorization errors."""

    def __init__(
        self,
        reason: str,
        *,
        user_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | int | None = None,
        remediation: str | None = None,
    ):
        self.user_id = user_id
        self.resource_type = resource_type
        self.resource_id = resource_id

        message = f"Security error: {reason}"
        if user_id:
            message += f" (user: {user_id})"
        if resource_type and resource_id:
            message += f" (resource: {resource_type}/{resource_id})"

        if not remediation:
            remediation = "Check authentication credentials and permissions"

        super().__init__(message=message, remediation=remediation)


class TenantContextError(SecurityError):
    """Raised when RequestContext cannot be built (missing user/workspace)."""

    def __init__(
        self,
        reason: str,
        *,
        missing_field: str | None = None,
        provided_data: dict | None = None,
        remediation: str | None = None,
    ):
        self.missing_field = missing_field
        self.provided_data = provided_data

        message = f"Invalid tenant context: {reason}"
        if missing_field:
            message += f" (missing: {missing_field})"

        if not remediation:
            remediation = "Ensure user_id and workspace_id are provided in request"

        super().__init__(
            reason=reason,
            remediation=remediation,
        )


class WorkspaceAccessDenied(SecurityError):
    """Raised when user attempts to access data outside their workspace."""

    def __init__(
        self,
        user_id: str,
        workspace_id: str,
        resource_type: str,
        resource_id: str | int,
        *,
        reason: str | None = None,
        user_workspaces: list[int] | None = None,
        remediation: str | None = None,
    ):
        self.workspace_id = workspace_id
        self.user_workspaces = user_workspaces

        message = (
            f"Access denied to {resource_type} '{resource_id}' "
            f"for user {user_id} in workspace {workspace_id}"
        )
        if reason:
            message += f": {reason}"
        if user_workspaces:
            message += f" (user has access to workspaces: {user_workspaces})"

        if not remediation:
            remediation = "Check user workspace membership or request access"

        super().__init__(
            reason=message,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            remediation=remediation,
        )


class QuotaExceededError(SecurityError):
    """Raised when user reaches subscription or workspace limits."""

    def __init__(
        self,
        user_id: str,
        quota_type: str,
        current_usage: int,
        quota_limit: int,
        *,
        workspace_id: str | None = None,
        remediation: str | None = None,
    ):
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.quota_limit = quota_limit
        self.workspace_id = workspace_id

        message = (
            f"Quota exceeded for {quota_type}: "
            f"{current_usage}/{quota_limit} (user: {user_id}"
        )
        if workspace_id:
            message += f", workspace: {workspace_id}"
        message += ")"

        if not remediation:
            remediation = f"Upgrade your plan to increase {quota_type} quota"

        super().__init__(
            reason=message,
            user_id=user_id,
            remediation=remediation,
        )


# =============================================================================
# 3. Processing & Ingestion Exceptions (422/400)
# =============================================================================


class ProcessingError(FOSRAError):
    """Base for errors during document processing pipeline."""

    def __init__(
        self,
        operation: str,
        reason: str,
        *,
        source_id: str | None = None,
        file_path: str | None = None,
        remediation: str | None = None,
    ):
        self.operation = operation
        self.source_id = source_id
        self.file_path = file_path

        message = f"Processing error during {operation}: {reason}"
        if source_id:
            message += f" (source: {source_id})"
        elif file_path:
            message += f" (file: {file_path})"

        if not remediation:
            remediation = f"Check {operation} configuration and input data"

        super().__init__(message=message, remediation=remediation)


class ParsingError(ProcessingError):
    """Raised when document parsing fails."""

    def __init__(
        self,
        file_path: str,
        parser_type: str,
        reason: str,
        *,
        source_id: str | None = None,
        remediation: str | None = None,
    ):
        self.file_path = file_path
        self.parser_type = parser_type

        message = f"Failed to parse {file_path} using {parser_type}: {reason}"

        if not remediation:
            remediation = "Check document format and parser compatibility"

        super().__init__(
            operation="parsing",
            reason=message,
            source_id=source_id,
            file_path=file_path,
            remediation=remediation,
        )


class UnsupportedFileTypeError(ProcessingError):
    """Raised when file format is not supported by any parser."""

    def __init__(
        self,
        file_path: str,
        file_type: str,
        *,
        supported_types: list[str] | None = None,
        remediation: str | None = None,
    ):
        self.file_path = file_path
        self.file_type = file_type
        self.supported_types = supported_types

        message = f"Unsupported file type '{file_type}' for {file_path}"
        if supported_types:
            message += f". Supported types: {', '.join(supported_types)}"

        if not remediation:
            remediation = (
                "Convert file to a supported format or install appropriate parser"
            )

        super().__init__(
            operation="file_type_detection",
            reason=message,
            file_path=file_path,
            remediation=remediation,
        )


class EmbeddingError(ProcessingError):
    """Raised when embedding generation fails."""

    def __init__(
        self,
        text_preview: str,
        embedder_type: str,
        reason: str,
        *,
        text_length: int | None = None,
        source_id: str | None = None,
        remediation: str | None = None,
    ):
        self.text_preview = text_preview[:100]
        self.embedder_type = embedder_type
        self.text_length = text_length

        message = f"Failed to generate embedding using {embedder_type}: {reason}"
        if text_length:
            message += f" (text length: {text_length} chars)"

        if not remediation:
            remediation = "Check embedder configuration and text content"

        super().__init__(
            operation="embedding",
            reason=message,
            source_id=source_id,
            remediation=remediation,
        )


class ChunkingError(ProcessingError):
    """Raised when document chunking fails."""

    def __init__(
        self,
        source_id: str,
        chunker_type: str,
        reason: str,
        *,
        document_length: int | None = None,
        chunk_config: dict | None = None,
        remediation: str | None = None,
    ):
        self.chunker_type = chunker_type
        self.document_length = document_length
        self.chunk_config = chunk_config

        message = (
            f"Chunking failed for source {source_id} using {chunker_type}: {reason}"
        )
        if document_length:
            message += f" (document: {document_length} chars)"

        if not remediation:
            remediation = "Check chunker configuration and document format"

        super().__init__(
            operation="chunking",
            reason=message,
            source_id=source_id,
            remediation=remediation,
        )


class ContextWindowError(ProcessingError):
    """Raised when content exceeds LLM context window."""

    def __init__(
        self,
        model: str,
        content_tokens: int,
        max_tokens: int,
        *,
        operation: str = "llm_request",
        remediation: str | None = None,
    ):
        self.model = model
        self.content_tokens = content_tokens
        self.max_tokens = max_tokens

        message = (
            f"Content exceeds context window for {model}: "
            f"{content_tokens} tokens > {max_tokens} max"
        )

        if not remediation:
            remediation = (
                f"Reduce content size or use a model with larger context window "
                f"(current: {max_tokens} tokens)"
            )

        super().__init__(
            operation=operation,
            reason=message,
            remediation=remediation,
        )


# =============================================================================
# 4. Storage & Retrieval Exceptions (404/500)
# =============================================================================


class StorageError(FOSRAError):
    """Base for storage operation errors."""

    def __init__(
        self,
        operation: str,
        reason: str,
        *,
        storage_type: str | None = None,
        resource_id: str | None = None,
        remediation: str | None = None,
    ):
        self.operation = operation
        self.storage_type = storage_type
        self.resource_id = resource_id

        message = f"Storage error during {operation}: {reason}"
        if storage_type:
            message += f" (storage: {storage_type})"
        if resource_id:
            message += f" (resource: {resource_id})"

        if not remediation:
            remediation = (
                f"Check {storage_type or 'storage'} connectivity and configuration"
            )

        super().__init__(message=message, remediation=remediation)


class VectorStorageError(StorageError):
    """Raised when vector store operations fail."""

    def __init__(
        self,
        operation: str,
        reason: str,
        *,
        collection_name: str | None = None,
        point_id: str | None = None,
        remediation: str | None = None,
    ):
        self.collection_name = collection_name
        self.point_id = point_id

        message = f"Vector storage error during {operation}: {reason}"
        if collection_name:
            message += f" (collection: {collection_name})"
        if point_id:
            message += f" (point: {point_id})"

        if not remediation:
            remediation = "Check vector store connectivity and collection configuration"

        super().__init__(
            operation=operation,
            reason=message,
            storage_type="vector",
            remediation=remediation,
        )


class VectorRetrievalError(StorageError):
    """Raised when vector retrieval operations fail."""

    def __init__(
        self,
        query_text: str,
        query_vector: list[float],
        reason: str,
        *,
        collection_name: str | None = None,
        top_k: int | None = None,
        remediation: str | None = None,
    ):
        self.query = query_vector
        self.collection_name = collection_name
        self.top_k = top_k

        message = (
            f"Vector retrieval failed for query '{query_vector[:50]}...': {reason}"
        )
        if collection_name:
            message += f" (collection: {collection_name})"
        if top_k:
            message += f" (top_k: {top_k})"

        if not remediation:
            remediation = "Check vector store connectivity and query parameters"

        super().__init__(
            operation="vector_search",
            reason=message,
            storage_type="vector",
            remediation=remediation,
        )


class FileStorageError(StorageError):
    """Raised when file storage operations fail."""

    def __init__(
        self,
        operation: str,
        file_path: str,
        reason: str,
        *,
        backend_type: str | None = None,
        remediation: str | None = None,
    ):
        self.file_path = file_path
        self.backend_type = backend_type

        message = f"File storage error during {operation} for {file_path}: {reason}"
        if backend_type:
            message += f" (backend: {backend_type})"

        if not remediation:
            remediation = (
                f"Check file permissions and {backend_type or 'storage'} connectivity"
            )

        super().__init__(
            operation=operation,
            reason=message,
            storage_type="file",
            remediation=remediation,
        )


class DatabaseError(StorageError):
    """Base for database operation errors."""

    def __init__(
        self,
        operation: str,
        reason: str,
        *,
        table: str | None = None,
        record_id: str | None = None,
        remediation: str | None = None,
    ):
        self.table = table
        self.record_id = record_id

        message = f"Database error during {operation}: {reason}"
        if table:
            message += f" (table: {table})"
        if record_id:
            message += f" (record: {record_id})"

        if not remediation:
            remediation = "Check database connectivity and query syntax"

        super().__init__(
            operation=operation,
            reason=message,
            storage_type="database",
            remediation=remediation,
        )


class UserStorageError(DatabaseError):
    """Raised when user storage operations fail."""

    def __init__(
        self,
        operation: str,
        user_id: str,
        reason: str,
        *,
        remediation: str | None = None,
    ):
        self.user_id = user_id

        if not remediation:
            remediation = "Check user data format and database connectivity"

        super().__init__(
            operation=operation,
            reason=f"User storage failed for {user_id}: {reason}",
            table="users",
            record_id=user_id,
            remediation=remediation,
        )


class UserRetrievalError(DatabaseError):
    """Raised when user retrieval operations fail."""

    def __init__(
        self,
        user_id: str,
        reason: str,
        *,
        lookup_field: str | None = None,
        remediation: str | None = None,
    ):
        self.user_id = user_id
        self.lookup_field = lookup_field

        message = f"Failed to retrieve user {user_id}: {reason}"
        if lookup_field:
            message += f" (lookup: {lookup_field})"

        if not remediation:
            remediation = "Check user ID and database connectivity"

        super().__init__(
            operation="user_retrieval",
            reason=message,
            table="users",
            record_id=user_id,
            remediation=remediation,
        )


class UserExistenceError(DatabaseError):
    """Raised when user does not exist."""

    def __init__(
        self,
        user_id: str,
        *,
        lookup_field: str = "user_id",
        remediation: str | None = None,
    ):
        self.user_id = user_id
        self.lookup_field = lookup_field

        if not remediation:
            remediation = "Verify user ID or create user account"

        super().__init__(
            operation="user_lookup",
            reason=f"User not found: {user_id} (lookup: {lookup_field})",
            table="users",
            record_id=user_id,
            remediation=remediation,
        )


class WorkspaceStorageError(DatabaseError):
    """Raised when workspace storage operations fail."""

    def __init__(
        self,
        operation: str,
        workspace_id: str,
        reason: str,
        *,
        user_id: str | None = None,
        remediation: str | None = None,
    ):
        self.workspace_id = workspace_id
        self.user_id = user_id

        message = f"Workspace storage failed for workspace {workspace_id}: {reason}"
        if user_id:
            message += f" (user: {user_id})"

        if not remediation:
            remediation = "Check workspace data format and database connectivity"

        super().__init__(
            operation=operation,
            reason=message,
            table="workspaces",
            record_id=str(workspace_id),
            remediation=remediation,
        )


class WorkspaceRetrievalError(DatabaseError):
    """Raised when workspace retrieval operations fail."""

    def __init__(
        self,
        workspace_id: str,
        reason: str,
        *,
        user_id: str | None = None,
        remediation: str | None = None,
    ):
        self.workspace_id = workspace_id
        self.user_id = user_id

        message = f"Failed to retrieve workspace {workspace_id}: {reason}"
        if user_id:
            message += f" (user: {user_id})"

        if not remediation:
            remediation = "Check workspace ID and user permissions"

        super().__init__(
            operation="workspace_retrieval",
            reason=message,
            table="workspaces",
            record_id=workspace_id,
            remediation=remediation,
        )


class WorkspaceExistenceError(DatabaseError):
    """Raised when workspace does not exist."""

    def __init__(
        self,
        workspace_id: str,
        *,
        user_id: str | None = None,
        remediation: str | None = None,
    ):
        self.workspace_id = workspace_id
        self.user_id = user_id

        message = f"Workspace not found: {workspace_id}"
        if user_id:
            message += f" (user: {user_id})"

        if not remediation:
            remediation = "Verify workspace ID or create workspace"

        super().__init__(
            operation="workspace_lookup",
            reason=message,
            table="workspaces",
            record_id=str(workspace_id),
            remediation=remediation,
        )


class ConvoStorageError(DatabaseError):
    """Raised when convo storage operations fail."""

    def __init__(
        self,
        operation: str,
        convo_id: str,
        reason: str,
        *,
        user_id: str | None = None,
        remediation: str | None = None,
    ):
        self.convo_id = convo_id
        self.user_id = user_id

        message = f"Convo storage failed for {convo_id}: {reason}"
        if user_id:
            message += f" (user: {user_id})"

        if not remediation:
            remediation = "Check convo data format and database connectivity"

        super().__init__(
            operation=operation,
            reason=message,
            table="convos",
            record_id=convo_id,
            remediation=remediation,
        )


class ConvoRetrievalError(DatabaseError):
    """Raised when convo retrieval operations fail."""

    def __init__(
        self,
        convo_id: str,
        reason: str,
        *,
        user_id: str | None = None,
        remediation: str | None = None,
    ):
        self.convo_id = convo_id
        self.user_id = user_id

        message = f"Failed to retrieve convo {convo_id}: {reason}"
        if user_id:
            message += f" (user: {user_id})"

        if not remediation:
            remediation = "Check convo ID and user permissions"

        super().__init__(
            operation="convo_retrieval",
            reason=message,
            table="convos",
            record_id=convo_id,
            remediation=remediation,
        )


class CollectionNotFoundError(VectorStorageError):
    """Raised when vector collection does not exist."""

    def __init__(
        self,
        collection_name: str,
        *,
        vector_store: str | None = None,
        available_collections: list[str] | None = None,
        remediation: str | None = None,
    ):
        self.available_collections = available_collections

        message = f"Vector collection not found: {collection_name}"
        if vector_store:
            message += f" (store: {vector_store})"
        if available_collections:
            message += f". Available: {', '.join(available_collections)}"

        if not remediation:
            remediation = (
                f"Create collection '{collection_name}' or use an existing collection"
            )

        super().__init__(
            operation="collection_lookup",
            reason=message,
            collection_name=collection_name,
            remediation=remediation,
        )


class EmptyRetrievalError(VectorRetrievalError):
    """Raised when no results are found (can trigger fallbacks)."""

    def __init__(
        self,
        query_vector: list[float],
        query_text: str | None = None,
        *,
        collection_name: str | None = None,
        filters: dict | None = None,
        remediation: str | None = None,
    ):
        self.filters = filters

        message = f"No results found for query '{query_vector[:50]}...'"
        if collection_name:
            message += f" in collection '{collection_name}'"
        if filters:
            message += f" with filters: {filters}"

        if not remediation:
            remediation = "Try broader search terms or adjust filters"

        super().__init__(
            query_text="",
            query_vector=query_vector,
            reason=message,
            collection_name=collection_name,
            remediation=remediation,
        )


# =============================================================================
# 5. Convo & Streaming Exceptions (400/500)
# =============================================================================


class ConvoError(FOSRAError):
    """Base for convo and chat errors."""

    def __init__(
        self,
        reason: str,
        *,
        convo_id: str | None = None,
        user_id: str | None = None,
        remediation: str | None = None,
    ):
        self.convo_id = convo_id
        self.user_id = user_id

        message = f"Convo error: {reason}"
        if convo_id:
            message += f" (convo: {convo_id})"
        if user_id:
            message += f" (user: {user_id})"

        if not remediation:
            remediation = "Check convo state and configuration"

        super().__init__(message=message, remediation=remediation)


class SessionNotFoundError(ConvoError):
    """Raised when convo session does not exist."""

    def __init__(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        remediation: str | None = None,
    ):
        self.session_id = session_id

        if not remediation:
            remediation = "Create a new session or verify session ID"

        super().__init__(
            reason=f"Session not found: {session_id}",
            convo_id=session_id,
            user_id=user_id,
            remediation=remediation,
        )


class StateTransitionError(ConvoError):
    """Raised when invalid state transition occurs in LangGraph."""

    def __init__(
        self,
        from_state: str,
        to_state: str,
        reason: str,
        *,
        convo_id: str | None = None,
        allowed_transitions: list[str] | None = None,
        remediation: str | None = None,
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.allowed_transitions = allowed_transitions

        message = (
            f"Invalid state transition from '{from_state}' to '{to_state}': {reason}"
        )
        if allowed_transitions:
            message += f". Allowed: {', '.join(allowed_transitions)}"

        if not remediation:
            remediation = "Check convo flow and state machine configuration"

        super().__init__(
            reason=message,
            convo_id=convo_id,
            remediation=remediation,
        )


class ConvoStateError(ConvoError):
    """Raised when convo state is invalid or initialization fails."""

    def __init__(
        self,
        reason: str,
        *,
        convo_id: str | None = None,
        state_data: dict | None = None,
        remediation: str | None = None,
    ):
        self.state_data = state_data

        if not remediation:
            remediation = "Reinitialize convo state or check state data format"

        super().__init__(
            reason=f"Invalid convo state: {reason}",
            convo_id=convo_id,
            remediation=remediation,
        )


class ConvoValidationError(ConvoError):
    """Raised when convo validation fails."""

    def __init__(
        self,
        field: str,
        reason: str,
        *,
        convo_id: str | None = None,
        provided_value: Any | None = None,
        remediation: str | None = None,
    ):
        self.field = field
        self.provided_value = provided_value

        message = f"Convo validation failed for '{field}': {reason}"
        if provided_value is not None:
            message += f" (provided: {provided_value})"

        if not remediation:
            remediation = f"Provide valid value for {field}"

        super().__init__(
            reason=message,
            convo_id=convo_id,
            remediation=remediation,
        )


class StreamingError(FOSRAError):
    """Base for streaming-related errors."""

    def __init__(
        self,
        reason: str,
        *,
        stream_id: str | None = None,
        operation: str | None = None,
        remediation: str | None = None,
    ):
        self.stream_id = stream_id
        self.operation = operation

        message = f"Streaming error: {reason}"
        if stream_id:
            message += f" (stream: {stream_id})"
        if operation:
            message += f" (operation: {operation})"

        if not remediation:
            remediation = "Check streaming connection and network connectivity"

        super().__init__(message=message, remediation=remediation)


class StreamConnectionError(StreamingError):
    """Raised when streaming connection fails or is interrupted."""

    def __init__(
        self,
        reason: str,
        *,
        stream_id: str | None = None,
        endpoint: str | None = None,
        remediation: str | None = None,
    ):
        self.endpoint = endpoint

        message = f"Stream connection failed: {reason}"
        if endpoint:
            message += f" (endpoint: {endpoint})"

        if not remediation:
            remediation = "Check network connectivity and retry streaming"

        super().__init__(
            reason=message,
            stream_id=stream_id,
            operation="connection",
            remediation=remediation,
        )


class StreamProcessingError(StreamingError):
    """Raised when stream chunk processing fails."""

    def __init__(
        self,
        reason: str,
        *,
        stream_id: str | None = None,
        chunk_index: int | None = None,
        chunk_data: Any | None = None,
        remediation: str | None = None,
    ):
        self.chunk_index = chunk_index
        self.chunk_data = chunk_data

        message = f"Stream chunk processing failed: {reason}"
        if chunk_index is not None:
            message += f" (chunk: {chunk_index})"

        if not remediation:
            remediation = "Check chunk format and processing logic"

        super().__init__(
            reason=message,
            stream_id=stream_id,
            operation="chunk_processing",
            remediation=remediation,
        )


# =============================================================================
# 6. LLM Operations Exceptions (500/503)
# =============================================================================


class LLMError(FOSRAError):
    """Base for LLM-related errors."""

    def __init__(
        self,
        reason: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        operation: str | None = None,
        remediation: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.operation = operation

        message = f"LLM error: {reason}"
        if provider and model:
            message += f" ({provider}/{model})"
        if operation:
            message += f" during {operation}"

        if not remediation:
            remediation = "Check LLM configuration and API connectivity"

        super().__init__(message=message, remediation=remediation)


class LLMValidationError(LLMError):
    """Raised when LLM configuration validation fails."""

    def __init__(
        self,
        provider: str,
        model: str,
        reason: str,
        *,
        invalid_field: str | None = None,
        remediation: str | None = None,
    ):
        self.invalid_field = invalid_field

        message = f"LLM validation failed for {provider}/{model}: {reason}"
        if invalid_field:
            message += f" (field: {invalid_field})"

        if not remediation:
            remediation = "Check API key, model name, and provider configuration"

        super().__init__(
            reason=message,
            provider=provider,
            model=model,
            operation="validation",
            remediation=remediation,
        )


class LLMConfigurationError(LLMError):
    """Raised when LLM configuration is invalid or missing."""

    def __init__(
        self,
        reason: str,
        *,
        llm_role: str | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
        remediation: str | None = None,
    ):
        self.llm_role = llm_role
        self.user_id = user_id
        self.workspace_id = workspace_id

        message = f"LLM configuration error: {reason}"
        if llm_role:
            message += f" (role: {llm_role})"
        if user_id:
            message += f" (user: {user_id})"
        if workspace_id:
            message += f" (workspace: {workspace_id})"

        if not remediation:
            remediation = "Provide valid LLM configuration in user preferences"

        super().__init__(
            reason=message,
            operation="configuration",
            remediation=remediation,
        )


# =============================================================================
# 7. Embedder Exceptions
# =============================================================================


class EmbedderError(FOSRAError):
    """Base for embedder-related errors."""

    def __init__(
        self,
        reason: str,
        *,
        embedder_type: str | None = None,
        operation: str | None = None,
        remediation: str | None = None,
    ):
        self.embedder_type = embedder_type
        self.operation = operation

        message = f"Embedder error: {reason}"
        if embedder_type:
            message += f" ({embedder_type})"
        if operation:
            message += f" during {operation}"

        if not remediation:
            remediation = "Check embedder configuration and dependencies"

        super().__init__(message=message, remediation=remediation)


class EmbedderNotFoundError(EmbedderError):
    """Raised when embedder type is not available."""

    def __init__(
        self,
        embedder_type: str,
        *,
        available_embedders: list[str] | None = None,
        remediation: str | None = None,
    ):
        self.available_embedders = available_embedders

        message = f"Embedder not found: {embedder_type}"
        if available_embedders:
            message += f". Available: {', '.join(available_embedders)}"

        if not remediation:
            remediation = "Check that the embedder type is registered and available"

        super().__init__(
            reason=message,
            embedder_type=embedder_type,
            operation="lookup",
            remediation=remediation,
        )


class EmbedderInitializationError(EmbedderError):
    """Raised when embedder fails to initialize."""

    def __init__(
        self,
        embedder_type: str,
        reason: str,
        *,
        model_name: str | None = None,
        remediation: str | None = None,
    ):
        self.model_name = model_name

        message = f"Failed to initialize {embedder_type}"
        if model_name:
            message += f" (model: {model_name})"
        message += f": {reason}"

        if not remediation:
            remediation = "Check model availability and configuration"

        super().__init__(
            reason=message,
            embedder_type=embedder_type,
            operation="initialization",
            remediation=remediation,
        )


class EmbeddingOperationError(EmbedderError):
    """Raised when embedding operation fails."""

    def __init__(
        self,
        operation: str,
        reason: str,
        *,
        embedder_type: str | None = None,
        text_count: int | None = None,
        remediation: str | None = None,
    ):
        self.text_count = text_count

        message = f"Embedding operation '{operation}' failed: {reason}"
        if text_count:
            message += f" ({text_count} texts)"

        if not remediation:
            remediation = "Check input text and embedder configuration"

        super().__init__(
            reason=message,
            embedder_type=embedder_type,
            operation=operation,
            remediation=remediation,
        )


class APIKeyMissingError(EmbedderError):
    """Raised when API key is required but not provided."""

    def __init__(
        self,
        embedder_type: str,
        *,
        key_name: str | None = None,
        remediation: str | None = None,
    ):
        self.key_name = key_name

        message = f"API key required for {embedder_type}"
        if key_name:
            message += f" ({key_name})"

        if not remediation:
            remediation = "Provide API key in embedder configuration"

        super().__init__(
            reason=message,
            embedder_type=embedder_type,
            operation="authentication",
            remediation=remediation,
        )


class ModelNotInitializedError(EmbedderError):
    """Raised when attempting to use uninitialized model."""

    def __init__(
        self,
        embedder_type: str,
        *,
        operation: str | None = None,
        remediation: str | None = None,
    ):
        message = f"Model not initialized for {embedder_type}"
        if operation:
            message += f" (attempted: {operation})"

        if not remediation:
            remediation = "Ensure embedder is initialized before use"

        super().__init__(
            reason=message,
            embedder_type=embedder_type,
            operation=operation or "use",
            remediation=remediation,
        )


# =============================================================================
# 8. Parser Exceptions
# =============================================================================


class ParserError(FOSRAError):
    """Base for parser-related errors."""

    def __init__(
        self,
        reason: str,
        *,
        parser_type: str | None = None,
        file_path: str | None = None,
        operation: str | None = None,
        remediation: str | None = None,
    ):
        self.parser_type = parser_type
        self.file_path = file_path
        self.operation = operation

        message = f"Parser error: {reason}"
        if parser_type:
            message += f" ({parser_type})"
        if file_path:
            message += f" for {file_path}"
        if operation:
            message += f" during {operation}"

        if not remediation:
            remediation = "Check parser configuration and document format"

        super().__init__(message=message, remediation=remediation)


class ParserNotFoundError(ParserError):
    """Raised when parser type is not available."""

    def __init__(
        self,
        parser_type: str,
        *,
        available_parsers: list[str] | None = None,
        remediation: str | None = None,
    ):
        self.available_parsers = available_parsers

        message = f"Parser not found: {parser_type}"
        if available_parsers:
            message += f". Available: {', '.join(available_parsers)}"

        if not remediation:
            remediation = "Check that the parser type is registered and available"

        super().__init__(
            reason=message,
            parser_type=parser_type,
            operation="lookup",
            remediation=remediation,
        )


class ParserInitializationError(ParserError):
    """Raised when parser fails to initialize."""

    def __init__(
        self,
        parser_type: str,
        reason: str,
        *,
        missing_dependency: str | None = None,
        remediation: str | None = None,
    ):
        self.missing_dependency = missing_dependency

        message = f"Failed to initialize {parser_type}: {reason}"
        if missing_dependency:
            message += f" (missing: {missing_dependency})"

        if not remediation:
            if missing_dependency:
                remediation = (
                    f"Install required dependency: pip install {missing_dependency}"
                )
            else:
                remediation = "Check parser dependencies and configuration"

        super().__init__(
            reason=message,
            parser_type=parser_type,
            operation="initialization",
            remediation=remediation,
        )


class ParsingOperationError(ParserError):
    """Raised when parsing operation fails."""

    def __init__(
        self,
        parser_type: str,
        file_path: str,
        reason: str,
        *,
        page_number: int | None = None,
        remediation: str | None = None,
    ):
        self.page_number = page_number

        message = f"Parsing failed for {file_path} using {parser_type}: {reason}"
        if page_number:
            message += f" (page: {page_number})"

        if not remediation:
            remediation = "Check document format and parser compatibility"

        super().__init__(
            reason=message,
            parser_type=parser_type,
            file_path=file_path,
            operation="parsing",
            remediation=remediation,
        )


class ParserTimeoutError(ParserError):
    """Raised when parsing operation times out."""

    def __init__(
        self,
        parser_type: str,
        file_path: str,
        timeout_seconds: int,
        *,
        document_size: int | None = None,
        remediation: str | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.document_size = document_size

        message = f"Parser {parser_type} timed out after {timeout_seconds}s while parsing {file_path}"
        if document_size:
            message += f" (size: {document_size} bytes)"

        if not remediation:
            remediation = "Increase timeout or use a faster parser"

        super().__init__(
            reason=message,
            parser_type=parser_type,
            file_path=file_path,
            operation="parsing",
            remediation=remediation,
        )


class UnsupportedDocumentTypeError(ParserError):
    """Raised when no parser supports the document type."""

    def __init__(
        self,
        document_type: str,
        file_path: str,
        *,
        available_parsers: list[str] | None = None,
        remediation: str | None = None,
    ):
        self.document_type = document_type
        self.available_parsers = available_parsers

        message = f"No parser available for document type {document_type}: {file_path}"
        if available_parsers:
            message += f". Available parsers: {', '.join(available_parsers)}"

        if not remediation:
            remediation = "Use a supported document type or install appropriate parser"

        super().__init__(
            reason=message,
            file_path=file_path,
            operation="type_detection",
            remediation=remediation,
        )


class ParserDependencyError(ParserError):
    """Raised when required parser dependency is missing."""

    def __init__(
        self,
        parser_type: str,
        dependency: str,
        *,
        install_command: str | None = None,
        remediation: str | None = None,
    ):
        self.dependency = dependency
        self.install_command = install_command

        message = f"Parser {parser_type} requires {dependency} which is not installed"

        if not remediation:
            if install_command:
                remediation = f"Install required dependency: {install_command}"
            else:
                remediation = f"Install required dependency: pip install {dependency}"

        super().__init__(
            reason=message,
            parser_type=parser_type,
            operation="dependency_check",
            remediation=remediation,
        )


class InvalidDocumentError(ParserError):
    """Raised when document is invalid or corrupted."""

    def __init__(
        self,
        file_path: str,
        reason: str,
        *,
        parser_type: str | None = None,
        validation_error: str | None = None,
        remediation: str | None = None,
    ):
        self.validation_error = validation_error

        message = f"Invalid document {file_path}: {reason}"
        if validation_error:
            message += f" ({validation_error})"

        if not remediation:
            remediation = "Check that the document is valid and not corrupted"

        super().__init__(
            reason=message,
            parser_type=parser_type,
            file_path=file_path,
            operation="validation",
            remediation=remediation,
        )


# =============================================================================
# 9. File Storage Backend Exceptions
# =============================================================================


class StorageBackendNotFoundError(FileStorageError):
    """Raised when storage backend is not available."""

    def __init__(
        self,
        backend_type: str,
        *,
        available_backends: list[str] | None = None,
        remediation: str | None = None,
    ):
        self.backend_type = backend_type
        self.available_backends = available_backends

        message = f"Storage backend not found: {backend_type}"
        if available_backends:
            message += f". Available: {', '.join(available_backends)}"

        if not remediation:
            remediation = "Check that the storage backend is registered and available"

        super().__init__(
            operation="backend_lookup",
            file_path="",
            reason=message,
            backend_type=backend_type,
            remediation=remediation,
        )


class StorageBackendInitializationError(FileStorageError):
    """Raised when storage backend fails to initialize."""

    def __init__(
        self,
        backend_type: str,
        reason: str,
        *,
        config_error: str | None = None,
        remediation: str | None = None,
    ):
        self.config_error = config_error

        message = f"Failed to initialize {backend_type} backend: {reason}"
        if config_error:
            message += f" ({config_error})"

        if not remediation:
            remediation = "Check backend configuration and credentials"

        super().__init__(
            operation="backend_initialization",
            file_path="",
            reason=message,
            backend_type=backend_type,
            remediation=remediation,
        )


class FileNotFoundError(FileStorageError):
    """Raised when file does not exist at specified path."""

    def __init__(
        self,
        file_path: str,
        *,
        backend_type: str | None = None,
        searched_locations: list[str] | None = None,
        remediation: str | None = None,
    ):
        self.searched_locations = searched_locations

        message = f"File not found: {file_path}"
        if searched_locations:
            message += f" (searched: {', '.join(searched_locations)})"

        if not remediation:
            remediation = "Verify the file path is correct and the file exists"

        super().__init__(
            operation="file_lookup",
            file_path=file_path,
            reason=message,
            backend_type=backend_type,
            remediation=remediation,
        )


class FileReadError(FileStorageError):
    """Raised when file reading operation fails."""

    def __init__(
        self,
        file_path: str,
        reason: str,
        *,
        backend_type: str | None = None,
        bytes_read: int | None = None,
        remediation: str | None = None,
    ):
        self.bytes_read = bytes_read

        message = f"Failed to read file {file_path}: {reason}"
        if bytes_read is not None:
            message += f" (read {bytes_read} bytes before failure)"

        if not remediation:
            remediation = "Check file permissions and storage backend connectivity"

        super().__init__(
            operation="file_read",
            file_path=file_path,
            reason=message,
            backend_type=backend_type,
            remediation=remediation,
        )


class FileListError(FileStorageError):
    """Raised when listing files fails."""

    def __init__(
        self,
        path: str,
        reason: str,
        *,
        backend_type: str | None = None,
        pattern: str | None = None,
        remediation: str | None = None,
    ):
        self.pattern = pattern

        message = f"Failed to list files at {path}: {reason}"
        if pattern:
            message += f" (pattern: {pattern})"

        if not remediation:
            remediation = "Check path and storage backend connectivity"

        super().__init__(
            operation="file_list",
            file_path=path,
            reason=message,
            backend_type=backend_type,
            remediation=remediation,
        )


class InvalidPathError(FileStorageError):
    """Raised when file path is invalid."""

    def __init__(
        self,
        file_path: str,
        reason: str,
        *,
        expected_format: str | None = None,
        remediation: str | None = None,
    ):
        self.expected_format = expected_format

        message = f"Invalid file path {file_path}: {reason}"
        if expected_format:
            message += f" (expected format: {expected_format})"

        if not remediation:
            remediation = "Provide a valid file path"

        super().__init__(
            operation="path_validation",
            file_path=file_path,
            reason=message,
            remediation=remediation,
        )


class UnsupportedBackendError(FileStorageError):
    """Raised when no backend supports the path type."""

    def __init__(
        self,
        file_path: str,
        *,
        available_backends: list[str] | None = None,
        path_scheme: str | None = None,
        remediation: str | None = None,
    ):
        self.available_backends = available_backends
        self.path_scheme = path_scheme

        message = f"No storage backend available for path: {file_path}"
        if path_scheme:
            message += f" (scheme: {path_scheme})"
        if available_backends:
            message += f". Available backends: {', '.join(available_backends)}"

        if not remediation:
            remediation = "Use a supported storage backend type"

        super().__init__(
            operation="backend_selection",
            file_path=file_path,
            reason=message,
            remediation=remediation,
        )


class StorageCredentialsError(FileStorageError):
    """Raised when storage credentials are missing or invalid."""

    def __init__(
        self,
        backend_type: str,
        reason: str,
        *,
        credential_type: str | None = None,
        remediation: str | None = None,
    ):
        self.credential_type = credential_type

        message = f"Invalid credentials for {backend_type}: {reason}"
        if credential_type:
            message += f" (credential: {credential_type})"

        if not remediation:
            remediation = "Provide valid credentials in backend configuration"

        super().__init__(
            operation="authentication",
            file_path="",
            reason=message,
            backend_type=backend_type,
            remediation=remediation,
        )


class StorageConnectionError(FileStorageError):
    """Raised when connection to storage backend fails."""

    def __init__(
        self,
        backend_type: str,
        reason: str,
        *,
        endpoint: str | None = None,
        timeout_seconds: int | None = None,
        remediation: str | None = None,
    ):
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

        message = f"Failed to connect to {backend_type}: {reason}"
        if endpoint:
            message += f" (endpoint: {endpoint})"
        if timeout_seconds:
            message += f" (timeout: {timeout_seconds}s)"

        if not remediation:
            remediation = "Check network connectivity and service availability"

        super().__init__(
            operation="connection",
            file_path="",
            reason=message,
            backend_type=backend_type,
            remediation=remediation,
        )


# =============================================================================
# 10. Reranker Exceptions
# =============================================================================


class RerankerError(FOSRAError):
    """Base for reranker-related errors."""

    def __init__(
        self,
        reason: str,
        *,
        reranker_type: str | None = None,
        operation: str | None = None,
        remediation: str | None = None,
    ):
        self.reranker_type = reranker_type
        self.operation = operation

        message = f"Reranker error: {reason}"
        if reranker_type:
            message += f" ({reranker_type})"
        if operation:
            message += f" during {operation}"

        if not remediation:
            remediation = "Check reranker configuration and dependencies"

        super().__init__(message=message, remediation=remediation)


class RerankerNotFoundError(RerankerError):
    """Raised when reranker is not registered or available."""

    def __init__(
        self,
        reranker_type: str,
        *,
        available_rerankers: list[str] | None = None,
        remediation: str | None = None,
    ):
        self.available_rerankers = available_rerankers

        message = f"Reranker not found: {reranker_type}"
        if available_rerankers:
            message += f". Available: {', '.join(available_rerankers)}"

        if not remediation:
            remediation = "Check that the reranker type is registered and available"

        super().__init__(
            reason=message,
            reranker_type=reranker_type,
            operation="lookup",
            remediation=remediation,
        )


class RerankerInitializationError(RerankerError):
    """Raised when reranker fails to initialize."""

    def __init__(
        self,
        reranker_type: str,
        reason: str,
        *,
        model_name: str | None = None,
        remediation: str | None = None,
    ):
        self.model_name = model_name

        message = f"Failed to initialize {reranker_type} reranker: {reason}"
        if model_name:
            message += f" (model: {model_name})"

        if not remediation:
            remediation = "Check reranker dependencies and configuration"

        super().__init__(
            reason=message,
            reranker_type=reranker_type,
            operation="initialization",
            remediation=remediation,
        )


class RerankerAPIConfigError(RerankerError):
    """Raised when users API Config fails."""

    def __init__(
        self,
        reranker_type: str,
        reason: str,
        user_id: str,
        config_id: str,
        *,
        remediation: str | None = None,
    ):
        message = f"Failed to initialize {reranker_type} reranker: {reason}"

        if not remediation:
            remediation = "Check reranker dependencies and configuration"

        super().__init__(
            reason=message,
            reranker_type=reranker_type,
            operation="initialization",
            remediation=remediation,
        )


class InvalidDocumentListError(RerankerError):
    """Raised when users API Config fails."""

    def __init__(
        self,
        reason: str,
        user_id: str | None,
        config_id: str,
        *,
        remediation: str | None = None,
    ):
        message = (
            f"Failed to initialize reranker due to invalid document list: {reason}"
        )

        if not remediation:
            remediation = "Check reranker dependencies and configuration"

        super().__init__(
            reason=message,
            operation="initialization",
            remediation=remediation,
        )


class RerankingOperationError(RerankerError):
    """Raised when reranking operation fails."""

    def __init__(
        self,
        reranker_type: str,
        reason: str,
        *,
        query: str | None = None,
        document_count: int | None = None,
        remediation: str | None = None,
    ):
        self.query = query
        self.document_count = document_count

        message = f"Reranking failed with {reranker_type}: {reason}"
        if document_count is not None:
            message += f" ({document_count} documents)"
        if query:
            message += f" for query '{query[:50]}...'"

        if not remediation:
            remediation = "Check reranker configuration and document format"

        super().__init__(
            reason=message,
            reranker_type=reranker_type,
            operation="reranking",
            remediation=remediation,
        )


class RerankerTimeoutError(RerankerError):
    """Raised when reranking operation times out."""

    def __init__(
        self,
        reranker_type: str,
        timeout_seconds: float,
        *,
        query: str | None = None,
        document_count: int | None = None,
        remediation: str | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.query = query
        self.document_count = document_count

        message = f"Reranking timed out after {timeout_seconds}s ({reranker_type})"
        if document_count is not None:
            message += f" with {document_count} documents"
        if query:
            message += f" for query '{query[:50]}...'"

        if not remediation:
            remediation = (
                "Increase timeout, reduce document count, or use faster reranker"
            )

        super().__init__(
            reason=message,
            reranker_type=reranker_type,
            operation="reranking",
            remediation=remediation,
        )


class RerankerDependencyError(RerankerError):
    """Raised when required reranker dependencies are missing."""

    def __init__(
        self,
        reranker_type: str,
        missing_dependency: str,
        *,
        install_command: str | None = None,
        remediation: str | None = None,
    ):
        self.missing_dependency = missing_dependency
        self.install_command = install_command

        message = f"Missing dependency for {reranker_type}: {missing_dependency}"

        if not remediation:
            if install_command:
                remediation = f"Install dependency: {install_command}"
            else:
                remediation = f"Install dependency: pip install {missing_dependency}"

        super().__init__(
            reason=message,
            reranker_type=reranker_type,
            operation="dependency_check",
            remediation=remediation,
        )


# =============================================================================
# 11. Query & Search Exceptions
# =============================================================================


class QueryError(FOSRAError):
    """Base for query-related errors."""

    def __init__(
        self,
        reason: str,
        *,
        query: str | None = None,
        operation: str | None = None,
        remediation: str | None = None,
    ):
        self.query = query
        self.operation = operation

        message = f"Query error: {reason}"
        if query:
            message += f" for query '{query[:50]}...'"
        if operation:
            message += f" during {operation}"

        if not remediation:
            remediation = "Check query format and parameters"

        super().__init__(message=message, remediation=remediation)


class QueryExpansionError(QueryError):
    """Raised when query expansion fails."""

    def __init__(
        self,
        query: str,
        strategy: str,
        reason: str,
        *,
        llm_error: str | None = None,
        remediation: str | None = None,
    ):
        self.strategy = strategy
        self.llm_error = llm_error

        message = f"Query expansion failed for '{query[:50]}...' using strategy '{strategy}': {reason}"
        if llm_error:
            message += f" (LLM error: {llm_error})"

        if not remediation:
            remediation = "Check LLM configuration and query text"

        super().__init__(
            reason=message,
            query=query,
            operation=f"expansion ({strategy})",
            remediation=remediation,
        )


class QueryReformulationError(QueryError):
    """Raised when query reformulation fails."""

    def __init__(
        self,
        query: str,
        strategy: str,
        reason: str,
        *,
        llm_error: str | None = None,
        remediation: str | None = None,
    ):
        self.strategy = strategy
        self.llm_error = llm_error

        message = f"Query reformulation failed for '{query[:50]}...' using strategy '{strategy}': {reason}"
        if llm_error:
            message += f" (LLM error: {llm_error})"

        if not remediation:
            remediation = "Check LLM configuration and query text"

        super().__init__(
            reason=message,
            query=query,
            operation=f"expansion ({strategy})",
            remediation=remediation,
        )


class RetrievalError(FOSRAError):
    """Base for retrieval-related errors."""

    pass


class VectorSearchError(RetrievalError):
    """Raised when vector search fails."""

    def __init__(
        self,
        query: str,
        reason: str,
        remediation: str = "Check vector store connectivity and configuration",
    ):
        super().__init__(
            message=f"Vector search failed for query '{query[:50]}...': {reason}",
            remediation=remediation,
        )


class ChunkRetrievalError(RetrievalError):
    """Raised when chunk retrieval fails."""

    def __init__(
        self,
        chunk_id: str,
        reason: str,
        remediation: str = "Check chunk ID and database connectivity",
    ):
        super().__init__(
            message=f"Failed to retrieve chunk {chunk_id}: {reason}",
            remediation=remediation,
        )


class SourceRetrievalError(RetrievalError):
    """Raised when source retrieval fails."""

    def __init__(
        self,
        source_id: str,
        reason: str,
        remediation: str = "Check source ID and database connectivity",
    ):
        super().__init__(
            message=f"Failed to retrieve source {source_id}: {reason}",
            remediation=remediation,
        )


class RerankingError(RetrievalError):
    """Raised when reranking operation fails."""

    def __init__(
        self,
        reason: str,
        result_count: int | None = None,
        remediation: str = "Check reranker configuration and connectivity",
    ):
        message = f"Reranking failed: {reason}"
        if result_count is not None:
            message += f" ({result_count} results)"

        super().__init__(
            message=message,
            remediation=remediation,
        )


class FusionError(RetrievalError):
    """Raised when result fusion fails."""

    def __init__(
        self,
        reason: str,
        result_lists_count: int | None = None,
        remediation: str = "Check result lists format and data",
    ):
        message = f"Result fusion failed: {reason}"
        if result_lists_count is not None:
            message += f" ({result_lists_count} result lists)"

        super().__init__(
            message=message,
            remediation=remediation,
        )


class MetadataFilterError(RetrievalError):
    """Raised when metadata filtering fails."""

    def __init__(
        self,
        filter_key: str,
        reason: str,
        remediation: str = "Check filter criteria format and chunk metadata",
    ):
        super().__init__(
            message=f"Metadata filter '{filter_key}' failed: {reason}",
            remediation=remediation,
        )


# =============================================================================
# 12. RAG-Specific Exceptions
# =============================================================================


class RAGError(FOSRAError):
    """Base for RAG (Retrieval-Augmented Generation) errors."""

    pass


class ContextRetrievalError(RAGError):
    """Raised when context retrieval fails during RAG."""

    def __init__(
        self,
        query: str,
        reason: str,
        remediation: str = "Check vector store and embedder configuration",
    ):
        super().__init__(
            message=f"Failed to retrieve context for query '{query[:50]}...': {reason}",
            remediation=remediation,
        )


class MessageStorageError(RAGError):
    """Raised when message storage fails."""

    def __init__(
        self,
        message_role: str,
        reason: str,
        remediation: str = "Check database connectivity and message data",
    ):
        super().__init__(
            message=f"Failed to store {message_role} message: {reason}",
            remediation=remediation,
        )


class ConfigStorageError(DatabaseError):
    """Raised when user storage operations fail."""

    def __init__(
        self,
        operation: str,
        user_id: str,
        config_id: str,
        reason: str,
        *,
        remediation: str | None = None,
    ):
        self.user_id = user_id

        if not remediation:
            remediation = "Check config data format and database connectivity"

        super().__init__(
            operation=operation,
            reason=f"Config storage failed for {config_id}: {reason}",
            table="users",
            remediation=remediation,
        )


class ConfigRetrievalError(DatabaseError):
    """Raised when user retrieval operations fail."""

    def __init__(
        self,
        config_id: str,
        user_id: str,
        reason: str,
        *,
        lookup_field: str | None = None,
        remediation: str | None = None,
    ):
        self.user_id = user_id
        self.lookup_field = lookup_field

        message = f"Failed to retrieve config {config_id}: {reason}"
        if lookup_field:
            message += f" (lookup: {lookup_field})"

        if not remediation:
            remediation = "Check user ID and database connectivity"

        super().__init__(
            operation="config_retrieval",
            reason=message,
            table="config",
            record_id=user_id,
            remediation=remediation,
        )


class ConfigExistenceError(DatabaseError):
    """Raised when config does not exist."""

    def __init__(
        self,
        user_id: str,
        config_id: str,
        *,
        lookup_field: str = "config_id",
        remediation: str | None = None,
    ):
        self.user_id = user_id
        self.lookup_field = lookup_field

        if not remediation:
            remediation = "Verify config ID or create config."

        super().__init__(
            operation="config_lookup",
            reason=f"User not found: {config_id} (lookup: {lookup_field})",
            table="config",
            record_id=user_id,
            remediation=remediation,
        )
