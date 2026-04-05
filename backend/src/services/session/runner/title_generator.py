"""title_generator — auto-generate conversation titles from user text."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from backend.src.services.session.event_emitter import EventEmitter

_TITLE_FILLER: frozenset[str] = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "so",
        "please",
        "can",
        "could",
        "would",
        "i",
        "want",
        "need",
        "help",
        "with",
        "my",
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
    }
)


async def maybe_generate_title(
    session_id: str,
    user_id: str,
    user_text: str,
    session_factory: async_sessionmaker[AsyncSession],
    emitter: EventEmitter,
    slog: Any,
) -> None:
    """Generate title from user_text if session title is still 'New Session'.

    Retrieves the session, checks if title == 'New Session', generates
    a title from the first 6 non-filler words, updates the DB, and emits
    a session_updated event.

    Does nothing if:
    - user_text is empty
    - session already has a non-default title
    """
    if not user_text:
        return

    async with session_factory() as db_session:
        from backend.src.api.schemas.session_api_schemas import SessionUpdateRequest
        from backend.src.services.session.conversation_service import (
            SessionService,
        )

        session_obj = await SessionService.get_session_by_id(
            session=db_session,
            user_id=user_id,
            session_id=session_id,
        )
        if session_obj.title != "New Session":
            return

        words = user_text.split()
        title_words = words[:6]
        title = " ".join(title_words)
        first_word = title_words[0].lower().rstrip(".,!?;:")
        if first_word in _TITLE_FILLER and len(title_words) > 1:
            for i, w in enumerate(title_words):
                if w.lower().rstrip(".,!?;:") not in _TITLE_FILLER:
                    title = " ".join(title_words[i:])
                    break
        if len(title) > 50:
            title = title[:47] + "..."
        title = title.rstrip(".,!?;:") or "New Session"

        _ = await SessionService.update_session(
            session=db_session,
            session_update=SessionUpdateRequest(
                user_id=user_id,
                session_id=session_id,
                title=title,
            ),
        )
        await emitter.emit_session_updated({"id": session_id, "title": title})
        slog.info("title_generated", new_title=title)
