"""
content_parser — parses AIMessageChunk content into ContentDelta list.

This module extracts content parsing logic from stream_consumer.py as a pure,
stateless/side-effect-free set of functions that return ContentDelta lists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessageChunk
from loguru import logger as log

from backend.src.services.processing.loader_service import ulid_factory


@dataclass
class ParseState:
    """mutable state for parsing a stream of chunks"""

    inside_think_block: bool = False
    reasoning_part_id: str | None = None
    reasoning_start_time: int | None = None
    full_reasoning_text: str = ""
    full_text: str = ""


@dataclass
class ContentDelta:
    """a delta of content from parsing a chunk"""

    kind: str
    content: str
    reasoning_part_id: str | None = None
    reasoning_start_time: int | None = None


def parse_additional_kwargs(
    msg: AIMessageChunk, state: ParseState
) -> list[ContentDelta]:
    """parse reasoning content from additional_kwargs.

    some providers (ollama/qwen3) send reasoning via additional_kwargs
    instead of content blocks.
    """
    deltas: list[ContentDelta] = []
    additional = msg.additional_kwargs or {}

    reasoning_from_kwargs = additional.get("reasoning_content", "") or additional.get(
        "reasoning_details", ""
    )
    if reasoning_from_kwargs:
        now_ts = int(time.time())
        if state.reasoning_part_id is None:
            state.reasoning_part_id = ulid_factory()
            state.reasoning_start_time = now_ts
            deltas.append(
                ContentDelta(
                    kind="reasoning_start",
                    content="",
                    reasoning_part_id=state.reasoning_part_id,
                    reasoning_start_time=now_ts,
                )
            )
        state.inside_think_block = True
        state.full_reasoning_text += reasoning_from_kwargs
        deltas.append(
            ContentDelta(
                kind="reasoning_delta",
                content=reasoning_from_kwargs,
                reasoning_part_id=state.reasoning_part_id,
                reasoning_start_time=state.reasoning_start_time,
            )
        )

    return deltas


def parse_content_blocks(
    blocks: list[dict[str, Any]], state: ParseState
) -> list[ContentDelta]:
    """parse content blocks from a list (e.g., from msg.content).

    handles block types: "thinking", "reasoning", "text"
    """
    deltas: list[ContentDelta] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")

        log.bind(
            _structured={
                "block_type": block_type,
                "current_inside_think_block": state.inside_think_block,
            }
        ).debug("[PARSE CONTENT BLOCKS] processing_block")

        try:
            if block_type in ("thinking", "reasoning"):
                # "thinking" = anthropic, "reasoning" = other providers
                reasoning_text = (
                    block.get("thinking", "") or block.get("reasoning", "") or ""
                )

                if reasoning_text:
                    now_ts = int(time.time())
                    if state.reasoning_part_id is None:
                        state.reasoning_part_id = ulid_factory()
                        state.reasoning_start_time = now_ts
                        deltas.append(
                            ContentDelta(
                                kind="reasoning_start",
                                content="",
                                reasoning_part_id=state.reasoning_part_id,
                                reasoning_start_time=now_ts,
                            )
                        )
                    state.full_reasoning_text += reasoning_text
                    deltas.append(
                        ContentDelta(
                            kind="reasoning_delta",
                            content=reasoning_text,
                            reasoning_part_id=state.reasoning_part_id,
                            reasoning_start_time=state.reasoning_start_time,
                        )
                    )
            elif block_type == "text":
                text_val = block.get("text", "") or ""
                if text_val:
                    state.full_text += text_val
                    deltas.append(ContentDelta(kind="text", content=text_val))
        except Exception as e:
            # re-raise with context — matches stream_consumer.py behavior
            raise RuntimeError(
                f"error_processing_content_block: block_type={block_type}"
            ) from e

    return deltas


def parse_string_content(content: str, state: ParseState) -> list[ContentDelta]:
    """parse string content with <think>... tag state machine.

    also handles /think ... /end_think line-level directives used by some
    reasoning models.
    """
    deltas: list[ContentDelta] = []
    remaining = content

    while remaining:
        if state.inside_think_block:
            # look for closing  or /end_think tag
            close_think_idx = remaining.find("")
            close_slash_idx = remaining.find("/end_think")
            if close_think_idx != -1 and (
                close_slash_idx == -1 or close_think_idx < close_slash_idx
            ):
                reasoning_chunk = remaining[:close_think_idx]
                remaining = remaining[close_think_idx + len("") :]
                state.inside_think_block = False
                if reasoning_chunk:
                    state.full_reasoning_text += reasoning_chunk
                    deltas.append(
                        ContentDelta(
                            kind="reasoning_delta",
                            content=reasoning_chunk,
                            reasoning_part_id=state.reasoning_part_id,
                            reasoning_start_time=state.reasoning_start_time,
                        )
                    )
            elif close_slash_idx != -1:
                reasoning_chunk = remaining[:close_slash_idx]
                remaining = remaining[close_slash_idx + len("/end_think") :]
                state.inside_think_block = False
                if reasoning_chunk:
                    state.full_reasoning_text += reasoning_chunk
                    deltas.append(
                        ContentDelta(
                            kind="reasoning_delta",
                            content=reasoning_chunk,
                            reasoning_part_id=state.reasoning_part_id,
                            reasoning_start_time=state.reasoning_start_time,
                        )
                    )
            else:
                # no closing tag — entire remaining is reasoning
                state.full_reasoning_text += remaining
                deltas.append(
                    ContentDelta(
                        kind="reasoning_delta",
                        content=remaining,
                        reasoning_part_id=state.reasoning_part_id,
                        reasoning_start_time=state.reasoning_start_time,
                    )
                )
                remaining = ""
        else:
            # look for opening <think> or /think tag
            open_think_idx = remaining.find("<think>")
            open_slash_idx = remaining.find("/think")
            if open_think_idx != -1 and (
                open_slash_idx == -1 or open_think_idx <= open_slash_idx
            ):
                text_before = remaining[:open_think_idx]
                remaining = remaining[open_think_idx + len("<think>") :]
                state.inside_think_block = True
                if text_before:
                    state.full_text += text_before
                    deltas.append(ContentDelta(kind="text", content=text_before))
                if state.reasoning_part_id is None:
                    now_ts = int(time.time())
                    state.reasoning_part_id = ulid_factory()
                    state.reasoning_start_time = now_ts
                    deltas.append(
                        ContentDelta(
                            kind="reasoning_start",
                            content="",
                            reasoning_part_id=state.reasoning_part_id,
                            reasoning_start_time=now_ts,
                        )
                    )
            elif open_slash_idx != -1:
                text_before = remaining[:open_slash_idx]
                remaining = remaining[open_slash_idx + len("/think") :]
                state.inside_think_block = True
                if text_before:
                    state.full_text += text_before
                    deltas.append(ContentDelta(kind="text", content=text_before))
                if state.reasoning_part_id is None:
                    now_ts = int(time.time())
                    state.reasoning_part_id = ulid_factory()
                    state.reasoning_start_time = now_ts
                    deltas.append(
                        ContentDelta(
                            kind="reasoning_start",
                            content="",
                            reasoning_part_id=state.reasoning_part_id,
                            reasoning_start_time=now_ts,
                        )
                    )
            else:
                # no opening tag — regular text
                state.full_text += remaining
                deltas.append(ContentDelta(kind="text", content=remaining))
                remaining = ""

    return deltas


def parse_chunk_content(msg: AIMessageChunk, state: ParseState) -> list[ContentDelta]:
    """top-level content parser — routes to appropriate parser based on content type.

    routes based on:
    - If content is a list → parse_content_blocks
    - If content is a string → parse_string_content
    """
    deltas: list[ContentDelta] = []

    log.bind(
        _structured={
            "msg_additional_kwargs": msg.additional_kwargs,
            "current_inside_think_block": state.inside_think_block,
        }
    ).debug("[PARSE CHUNK CONTENT] parsing_additional_kwargs")

    content = msg.content

    log.bind(
        _structured={
            "content_type": type(content).__name__,
            "content_preview": content[:100] if isinstance(content, str) else "N/A",
        }
    ).debug("[PARSE CHUNK CONTENT] parsing_content")

    if isinstance(content, list):
        dict_blocks = [b for b in content if isinstance(b, dict)]
        deltas.extend(parse_content_blocks(dict_blocks, state))
    elif isinstance(content, str) and content:
        deltas.extend(parse_string_content(content, state))

    return deltas
