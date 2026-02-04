from typing import Annotated
from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import StreamingResponse
from FOSRABack.src.api.dependencies import get_db_session
from FOSRABack.src.services.conversation.stream_service import generate_stream


router = APIRouter(prefix="/stream", tags=["Streaming"])


@router.post("/generate_stream")
async def inference(
    request: Request,
    text: Annotated[str, Body()],
    session=Depends(get_db_session),
):

    gen = StreamingResponse(
        content=generate_stream(text),
        media_type="text/event-stream",
    )
    return gen
