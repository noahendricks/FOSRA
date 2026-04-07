"""Characterization tests for EventFormatter — documents current behavior of each method."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.src.services.conversation.runner.event_formatter import EventFormatter


@pytest.fixture
def mock_emitter():
    return AsyncMock()


@pytest.fixture
def formatter(mock_emitter):
    return EventFormatter(
        emitter=mock_emitter,
        session_id="session-123",
        now=1000000,
        provider_id="test-provider",
        model_id="test-model",
    )


class TestEmitUserMessage:
    @pytest.mark.asyncio
    async def test_emit_user_message_calls_emit_message_updated(
        self, formatter, mock_emitter
    ):
        await formatter.emit_user_message(
            user_msg_id="user-1",
            agent_name="test-agent",
            user_text="Hello world",
        )
        mock_emitter.emit_message_updated.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_user_message_structure(self, formatter, mock_emitter):
        await formatter.emit_user_message(
            user_msg_id="user-1",
            agent_name="test-agent",
            user_text="Hello world",
        )
        call_args = mock_emitter.emit_message_updated.call_args[0][0]
        assert call_args["id"] == "user-1"
        assert call_args["sessionID"] == "session-123"
        assert call_args["role"] == "user"
        assert call_args["agent"] == "test-agent"
        assert call_args["model"]["providerID"] == "test-provider"
        assert call_args["model"]["modelID"] == "test-model"


class TestEmitUserTextPart:
    @pytest.mark.asyncio
    async def test_emit_user_text_part_calls_emit_message_part_updated(
        self, formatter, mock_emitter
    ):
        await formatter.emit_user_text_part(
            user_msg_id="user-1",
            text="some text",
        )
        mock_emitter.emit_message_part_updated.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_user_text_part_structure(self, formatter, mock_emitter):
        await formatter.emit_user_text_part(
            user_msg_id="user-1",
            text="some text",
        )
        call_args = mock_emitter.emit_message_part_updated.call_args[0][0]
        assert call_args["sessionID"] == "session-123"
        assert call_args["messageID"] == "user-1"
        assert call_args["type"] == "text"
        assert call_args["text"] == "some text"


class TestEmitAssistantMessageStart:
    @pytest.mark.asyncio
    async def test_emit_assistant_message_start_returns_text_part_id_and_step_start_id(
        self, formatter, mock_emitter
    ):
        text_part_id, step_start_id = await formatter.emit_assistant_message_start(
            assistant_msg_id="assistant-1",
            user_msg_id="user-1",
            agent_name="test-agent",
        )
        assert isinstance(text_part_id, str)
        assert isinstance(step_start_id, str)
        assert text_part_id != step_start_id

    @pytest.mark.asyncio
    async def test_emit_assistant_message_start_emits_three_events(
        self, formatter, mock_emitter
    ):
        await formatter.emit_assistant_message_start(
            assistant_msg_id="assistant-1",
            user_msg_id="user-1",
            agent_name="test-agent",
        )
        assert mock_emitter.emit_message_updated.call_count == 1
        assert mock_emitter.emit_message_part_updated.call_count == 2

    @pytest.mark.asyncio
    async def test_emit_assistant_message_start_message_structure(
        self, formatter, mock_emitter
    ):
        await formatter.emit_assistant_message_start(
            assistant_msg_id="assistant-1",
            user_msg_id="user-1",
            agent_name="test-agent",
        )
        msg_call_args = mock_emitter.emit_message_updated.call_args[0][0]
        assert msg_call_args["id"] == "assistant-1"
        assert msg_call_args["sessionID"] == "session-123"
        assert msg_call_args["role"] == "assistant"
        assert msg_call_args["parentID"] == "user-1"
        assert msg_call_args["modelID"] == "test-model"
        assert msg_call_args["providerID"] == "test-provider"
        assert msg_call_args["mode"] == "default"
        assert msg_call_args["agent"] == "test-agent"


class TestEmitReasoningStart:
    @pytest.mark.asyncio
    async def test_emit_reasoning_start_calls_emit_message_part_updated(
        self, formatter, mock_emitter
    ):
        await formatter.emit_reasoning_start(
            assistant_msg_id="assistant-1",
            reasoning_part_id="reasoning-1",
            reasoning_start_time=1000001,
        )
        mock_emitter.emit_message_part_updated.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_reasoning_start_structure(self, formatter, mock_emitter):
        await formatter.emit_reasoning_start(
            assistant_msg_id="assistant-1",
            reasoning_part_id="reasoning-1",
            reasoning_start_time=1000001,
        )
        call_args = mock_emitter.emit_message_part_updated.call_args[0][0]
        assert call_args["id"] == "reasoning-1"
        assert call_args["sessionID"] == "session-123"
        assert call_args["messageID"] == "assistant-1"
        assert call_args["type"] == "reasoning"
        assert call_args["text"] == ""


class TestEmitTextDelta:
    @pytest.mark.asyncio
    async def test_emit_text_delta_calls_emit_message_part_delta(
        self, formatter, mock_emitter
    ):
        await formatter.emit_text_delta(
            assistant_msg_id="assistant-1",
            part_id="text-part-1",
            text="hello",
        )
        mock_emitter.emit_message_part_delta.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_text_delta_passes_correct_args(self, formatter, mock_emitter):
        await formatter.emit_text_delta(
            assistant_msg_id="assistant-1",
            part_id="text-part-1",
            text="hello",
        )
        mock_emitter.emit_message_part_delta.assert_called_once_with(
            "session-123",
            "assistant-1",
            "text-part-1",
            "text",
            "hello",
        )


class TestEmitReasoningDelta:
    @pytest.mark.asyncio
    async def test_emit_reasoning_delta_calls_emit_message_part_delta_with_reasoning_type(
        self, formatter, mock_emitter
    ):
        await formatter.emit_reasoning_delta(
            assistant_msg_id="assistant-1",
            reasoning_part_id="reasoning-1",
            text="thinking...",
        )
        mock_emitter.emit_message_part_delta.assert_called_once_with(
            "session-123",
            "assistant-1",
            "reasoning-1",
            "text",
            "thinking...",
            part_type="reasoning",
        )


class TestEmitToolStart:
    @pytest.mark.asyncio
    async def test_emit_tool_start_returns_part_id(self, formatter, mock_emitter):
        part_id = await formatter.emit_tool_start(
            assistant_msg_id="assistant-1",
            call_id="call-1",
            tool_name="bash",
            tool_args={"command": "ls"},
        )
        assert isinstance(part_id, str)

    @pytest.mark.asyncio
    async def test_emit_tool_start_structure(self, formatter, mock_emitter):
        await formatter.emit_tool_start(
            assistant_msg_id="assistant-1",
            call_id="call-1",
            tool_name="bash",
            tool_args={"command": "ls"},
        )
        call_args = mock_emitter.emit_message_part_updated.call_args[0][0]
        assert call_args["sessionID"] == "session-123"
        assert call_args["messageID"] == "assistant-1"
        assert call_args["type"] == "tool"
        assert call_args["callID"] == "call-1"
        assert call_args["tool"] == "bash"
        assert call_args["state"]["status"] == "running"
        assert call_args["state"]["input"] == {"command": "ls"}


class TestEmitToolEnd:
    @pytest.mark.asyncio
    async def test_emit_tool_end_returns_new_part_id(self, formatter, mock_emitter):
        part_id = await formatter.emit_tool_end(
            assistant_msg_id="assistant-1",
            call_id="call-1",
            tool_name="bash",
            tool_args={"command": "ls"},
            output="file1.txt\nfile2.txt",
            start_time=1000000.0,
        )
        assert isinstance(part_id, str)

    @pytest.mark.asyncio
    async def test_emit_tool_end_structure(self, formatter, mock_emitter):
        await formatter.emit_tool_end(
            assistant_msg_id="assistant-1",
            call_id="call-1",
            tool_name="bash",
            tool_args={"command": "ls"},
            output="file1.txt\nfile2.txt",
            start_time=1000000.0,
        )
        call_args = mock_emitter.emit_message_part_updated.call_args[0][0]
        assert call_args["sessionID"] == "session-123"
        assert call_args["messageID"] == "assistant-1"
        assert call_args["type"] == "tool"
        assert call_args["callID"] == "call-1"
        assert call_args["tool"] == "bash"
        assert call_args["state"]["status"] == "completed"
        assert call_args["state"]["input"] == {"command": "ls"}
        assert call_args["state"]["output"] == "file1.txt\nfile2.txt"

    @pytest.mark.asyncio
    async def test_emit_tool_end_truncates_long_output(self, formatter, mock_emitter):
        long_output = "x" * 5000
        await formatter.emit_tool_end(
            assistant_msg_id="assistant-1",
            call_id="call-1",
            tool_name="bash",
            tool_args={},
            output=long_output,
            start_time=1000000.0,
        )
        call_args = mock_emitter.emit_message_part_updated.call_args[0][0]
        assert len(call_args["state"]["output"]) == 2000

    @pytest.mark.asyncio
    async def test_emit_tool_end_with_metadata(self, formatter, mock_emitter):
        metadata = {"exit_code": 0}
        await formatter.emit_tool_end(
            assistant_msg_id="assistant-1",
            call_id="call-1",
            tool_name="bash",
            tool_args={},
            output="success",
            start_time=1000000.0,
            metadata=metadata,
        )
        call_args = mock_emitter.emit_message_part_updated.call_args[0][0]
        assert call_args["state"]["metadata"] == metadata


class TestEmitStepFinish:
    @pytest.mark.asyncio
    async def test_emit_step_finish_calls_emit_message_part_updated(
        self, formatter, mock_emitter
    ):
        await formatter.emit_step_finish(
            assistant_msg_id="assistant-1",
            step_finish_id="finish-1",
        )
        mock_emitter.emit_message_part_updated.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_step_finish_default_reason(self, formatter, mock_emitter):
        await formatter.emit_step_finish(
            assistant_msg_id="assistant-1",
            step_finish_id="finish-1",
        )
        call_args = mock_emitter.emit_message_part_updated.call_args[0][0]
        assert call_args["type"] == "step-finish"
        assert call_args["reason"] == "stop"

    @pytest.mark.asyncio
    async def test_emit_step_finish_custom_reason(self, formatter, mock_emitter):
        await formatter.emit_step_finish(
            assistant_msg_id="assistant-1",
            step_finish_id="finish-1",
            reason="max_tokens",
        )
        call_args = mock_emitter.emit_message_part_updated.call_args[0][0]
        assert call_args["reason"] == "max_tokens"


class TestEmitTextFinal:
    @pytest.mark.asyncio
    async def test_emit_text_final_calls_emit_message_part_updated(
        self, formatter, mock_emitter
    ):
        await formatter.emit_text_final(
            assistant_msg_id="assistant-1",
            text_part_id="text-1",
            full_text="Final text",
            end_time=1000005,
        )
        mock_emitter.emit_message_part_updated.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_text_final_structure(self, formatter, mock_emitter):
        await formatter.emit_text_final(
            assistant_msg_id="assistant-1",
            text_part_id="text-1",
            full_text="Final text",
            end_time=1000005,
        )
        call_args = mock_emitter.emit_message_part_updated.call_args[0][0]
        assert call_args["id"] == "text-1"
        assert call_args["messageID"] == "assistant-1"
        assert call_args["type"] == "text"
        assert call_args["text"] == "Final text"
        assert call_args["time"]["start"] == 1000000
        assert call_args["time"]["end"] == 1000005


class TestEmitReasoningFinal:
    @pytest.mark.asyncio
    async def test_emit_reasoning_final_calls_emit_message_part_updated(
        self, formatter, mock_emitter
    ):
        await formatter.emit_reasoning_final(
            assistant_msg_id="assistant-1",
            reasoning_part_id="reasoning-1",
            reasoning_start_time=1000001,
            end_time=1000005,
        )
        mock_emitter.emit_message_part_updated.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_reasoning_final_structure(self, formatter, mock_emitter):
        await formatter.emit_reasoning_final(
            assistant_msg_id="assistant-1",
            reasoning_part_id="reasoning-1",
            reasoning_start_time=1000001,
            end_time=1000005,
            full_reasoning_text="Final reasoning",
        )
        call_args = mock_emitter.emit_message_part_updated.call_args[0][0]
        assert call_args["id"] == "reasoning-1"
        assert call_args["messageID"] == "assistant-1"
        assert call_args["type"] == "reasoning"
        assert call_args["text"] == "Final reasoning"
        assert call_args["time"]["start"] == 1000001
        assert call_args["time"]["end"] == 1000005


class TestEmitAssistantFinal:
    @pytest.mark.asyncio
    async def test_emit_assistant_final_calls_emit_message_updated(
        self, formatter, mock_emitter
    ):
        await formatter.emit_assistant_final(
            assistant_msg_id="assistant-1",
            user_msg_id="user-1",
            agent_name="test-agent",
            end_time=1000005,
        )
        mock_emitter.emit_message_updated.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_assistant_final_structure(self, formatter, mock_emitter):
        await formatter.emit_assistant_final(
            assistant_msg_id="assistant-1",
            user_msg_id="user-1",
            agent_name="test-agent",
            end_time=1000005,
        )
        call_args = mock_emitter.emit_message_updated.call_args[0][0]
        assert call_args["id"] == "assistant-1"
        assert call_args["sessionID"] == "session-123"
        assert call_args["role"] == "assistant"
        assert call_args["time"]["completed"] == 1000005
        assert call_args["parentID"] == "user-1"
        assert call_args["agent"] == "test-agent"
        assert call_args["finish"] == "stop"

    @pytest.mark.asyncio
    async def test_emit_assistant_final_with_error(self, formatter, mock_emitter):
        mock_error = MagicMock()
        mock_error.model_dump.return_value = {
            "type": "UnknownError",
            "message": "test error",
        }
        await formatter.emit_assistant_final(
            assistant_msg_id="assistant-1",
            user_msg_id="user-1",
            agent_name="test-agent",
            end_time=1000005,
            error=mock_error,
        )
        call_args = mock_emitter.emit_message_updated.call_args[0][0]
        assert call_args["error"] is not None


class TestEmitSessionError:
    @pytest.mark.asyncio
    async def test_emit_session_error_calls_emit_session_error(
        self, formatter, mock_emitter
    ):
        await formatter.emit_session_error("Something went wrong")
        mock_emitter.emit_session_error.assert_called_once_with(
            "session-123", "Something went wrong"
        )
