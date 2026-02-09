from __future__ import annotations

import json
from dataclasses import field
from enum import StrEnum
from typing import Any

from loguru import logger

from backend.src.storage.utils.converters import DomainStruct

__all__ = [
    # Enums
    "StreamType",
    "DataAnnotationType",
    # Data classes
    "TerminalMessage",
    "SourceNode",
    "TokenUsage",
    # Formatter
    "StreamingFormatter",
    # Factory
    "create_formatter",
    "get_shared_formatter",
]


# =============================================================================
# Protocol Constants
# =============================================================================


class StreamType(StrEnum):
    TEXT = "0"
    ERROR = "3"
    DATA = "8"
    COMPLETION = "d"


class DataAnnotationType(StrEnum):
    TERMINAL_INFO = "TERMINAL_INFO"
    SOURCES = "sources"
    ANSWER = "ANSWER"
    FURTHER_QUESTIONS = "FURTHER_QUESTIONS"
    STATUS = "STATUS"
    METADATA = "METADATA"


# =============================================================================
# Data Classes
# =============================================================================


class TerminalMessage(DomainStruct):
    id: int
    text: str
    type: str = "info"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "type": self.type}


class SourceNode(DomainStruct):
    id: str
    text: str
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "url": self.url,
            "metadata": self.metadata,
        }


class TokenUsage(DomainStruct):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used."""
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for JSON serialization."""
        return {
            "promptTokens": self.prompt_tokens,
            "completionTokens": self.completion_tokens,
            "totalTokens": self.total_tokens,
        }


# =============================================================================
# Streaming Formatter
# =============================================================================


class StreamingFormatter:
    def __init__(self, track_state: bool = True):
        self._track_state = track_state
        self._terminal_idx = 1
        self._terminal_info: list[TerminalMessage] = []
        self._sources: list[dict[str, Any]] = []
        self._answer_chunks: list[str] = []
        self._further_questions: list[dict[str, Any]] = []

        if track_state:
            logger.debug("StreamingFormatter initialized with state tracking")
        else:
            logger.debug("StreamingFormatter initialized without state tracking")

    @property
    def terminal_idx(self) -> int:
        return self._terminal_idx

    @terminal_idx.setter
    def terminal_idx(self, value: int) -> None:
        self._terminal_idx = value

    def reset(self) -> None:
        logger.debug("Resetting StreamingFormatter state")
        self._terminal_idx = 1
        self._terminal_info.clear()
        self._sources.clear()
        self._answer_chunks.clear()
        self._further_questions.clear()

    def get_accumulated_answer(self) -> str:
        return "".join(self._answer_chunks)

    def get_sources(self) -> list[dict[str, Any]]:
        return self._sources.copy()

    def get_terminal_info(self) -> list[TerminalMessage]:
        return self._terminal_info.copy()

    def get_further_questions(self) -> list[dict[str, Any]]:
        return self._further_questions.copy()

    def get_stats(self) -> dict[str, Any]:
        return {
            "terminal_messages": len(self._terminal_info),
            "sources": len(self._sources),
            "answer_length": len(self.get_accumulated_answer()),
            "further_questions": len(self._further_questions),
        }

    # =========================================================================
    # Formatting Methods
    # =========================================================================

    def format_terminal_info_delta(
        self,
        text: str,
        message_type: str = "info",
    ) -> str:
        message = TerminalMessage(
            id=self._terminal_idx,
            text=text,
            type=message_type,
        )
        self._terminal_idx += 1

        if self._track_state:
            self._terminal_info.append(message)

        annotation = {
            "type": DataAnnotationType.TERMINAL_INFO,
            "data": message.to_dict(),
        }

        logger.debug(f"Formatted terminal info: {message_type} - {text[:50]}...")
        return self._format_data_message(annotation)

    def format_sources_delta(self, sources: list[dict[str, Any]]) -> str:
        if self._track_state:
            self._sources = sources

        nodes: list[dict[str, Any]] = []
        for group in sources:
            for source in group.get("sources", []):
                node = SourceNode(
                    id=str(source.get("id", "")),
                    text=source.get("description", "").strip(),
                    url=source.get("url", ""),
                    metadata={
                        "title": source.get("title", ""),
                        "source_type": group.get("type", ""),
                        "group_name": group.get("name", ""),
                    },
                )
                nodes.append(node.to_dict())

        annotation = {
            "type": DataAnnotationType.SOURCES,
            "data": {"nodes": nodes},
        }

        logger.debug(f"Formatted {len(nodes)} source nodes")
        return self._format_data_message(annotation)

    def format_answer_delta(self, answer_chunk: str) -> str:
        if self._track_state:
            self._answer_chunks.append(answer_chunk)

        annotation = {
            "type": DataAnnotationType.ANSWER,
            "content": [answer_chunk],
        }

        return self._format_data_message(annotation)

    def format_further_questions_delta(
        self,
        further_questions: list[dict[str, Any]],
    ) -> str:
        if self._track_state:
            self._further_questions = further_questions

        questions_text = [
            q.get("question", "") for q in further_questions if q.get("question")
        ]

        annotation = {
            "type": DataAnnotationType.FURTHER_QUESTIONS,
            "data": questions_text,
        }

        logger.debug(f"Formatted {len(questions_text)} further questions")
        return self._format_data_message(annotation)

    def format_status_delta(
        self,
        status: str,
        progress: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        data: dict[str, Any] = {"status": status}

        if progress is not None:
            data["progress"] = progress

        if metadata:
            data["metadata"] = metadata

        annotation = {
            "type": DataAnnotationType.STATUS,
            "data": data,
        }

        logger.debug(f"Formatted status: {status}")
        return self._format_data_message(annotation)

    def format_text_chunk(self, text: str) -> str:
        return f"{StreamType.TEXT}:{json.dumps(text)}\n"

    def format_error(self, error_message: str) -> str:
        logger.warning(f"Formatting error message: {error_message[:100]}...")
        return f"{StreamType.ERROR}:{json.dumps(error_message)}\n"

    def format_completion(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        finish_reason: str = "stop",
    ) -> str:
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        completion_data = {
            "finishReason": finish_reason,
            "usage": usage.to_dict(),
        }

        logger.info(
            f"Formatting completion: {usage.total_tokens} total tokens, "
            f"reason: {finish_reason}"
        )
        return f"{StreamType.COMPLETION}:{json.dumps(completion_data)}\n"

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _format_data_message(self, annotation: dict[str, Any]) -> str:
        return f"{StreamType.DATA}:[{json.dumps(annotation)}]\n"

    def format_custom_data(
        self,
        data_type: str,
        data: Any,
    ) -> str:
        annotation = {"type": data_type, "data": data}
        logger.debug(f"Formatting custom data type: {data_type}")
        return self._format_data_message(annotation)


# =============================================================================
# Factory Functions
# =============================================================================


def create_formatter(track_state: bool = True) -> StreamingFormatter:
    return StreamingFormatter(track_state=track_state)


_shared_formatter: StreamingFormatter | None = None


def get_shared_formatter() -> StreamingFormatter:
    global _shared_formatter
    if _shared_formatter is None:
        _shared_formatter = StreamingFormatter(track_state=False)
        logger.debug("Created shared stateless formatter")
    return _shared_formatter


StreamingService = StreamingFormatter
create_streaming_service = create_formatter
get_formatter = get_shared_formatter
