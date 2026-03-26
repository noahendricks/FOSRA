"""
question routes: list pending question requests, reply to or reject a question.

the tui displays question prompts and the user replies via these endpoints.
reply/reject resolves the asyncio.Future that the agent is waiting on.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from backend.src.api.dependencies import get_current_user_id
from backend.src.api.events import event_bus
from backend.src.api.routes.oc.state import (
    pending_questions,
    question_requests,
)
from backend.src.api.schemas.tui_schemas import QuestionRequest

router = APIRouter(prefix="/oc/question", tags=["Question"])


@router.get("")
async def list_questions(
    user_id: Annotated[str, Depends(get_current_user_id)],
):
    """
    return all pending question requests across all sessions for the current user.
    the tui filters by sessionID on the client side.
    """
    all_requests: list[dict[str, Any]] = []
    for session_id, requests in question_requests.items():
        for req in requests:
            all_requests.append(req)
    return all_requests


@router.post("/{request_id}/reply")
async def reply_question(
    request_id: str,
    body: dict[str, Any],
):
    """
    reply to a question request with answers.
    body: { "sessionID": str, "answers": [["option1"], ["option2"]] }
    answers is a list of lists (one per question, selected option labels).
    resolves the pending asyncio.Future so the agent can continue.
    publishes question.replied event.
    """
    answers = body.get("answers", [])
    session_id = body.get("sessionID", "")

    future = pending_questions.get(request_id)
    if future is None:
        raise HTTPException(status_code=404, detail="Question request not found")

    for session_id_key, requests in list(question_requests.items()):
        question_requests[session_id_key] = [
            r for r in requests if r.get("id") != request_id
        ]

    if not future.done():
        future.set_result(answers)

    await event_bus.publish(
        {
            "type": "question.replied",
            "properties": {
                "sessionID": session_id,
                "requestID": request_id,
                "answers": answers,
            },
        }
    )
    return True


@router.post("/{request_id}/reject")
async def reject_question(
    request_id: str,
    body: dict[str, Any],
):
    """
    reject a question request.
    body: { "sessionID": str }
    resolves the pending asyncio.Future with 'reject' so the agent can handle it.
    publishes question.rejected event.
    """
    session_id = body.get("sessionID", "")

    future = pending_questions.get(request_id)
    if future is None:
        raise HTTPException(status_code=404, detail="Question request not found")

    for sid, requests in list(question_requests.items()):
        question_requests[sid] = [r for r in requests if r.get("id") != request_id]

    if not future.done():
        future.set_result("reject")

    await event_bus.publish(
        {
            "type": "question.rejected",
            "properties": {
                "sessionID": session_id,
                "requestID": request_id,
            },
        }
    )
    return True
