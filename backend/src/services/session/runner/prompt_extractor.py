from typing import Any

from backend.src.api.schemas.provider_registry import _build_providers
from backend.src.settings import LLMConfig


def extract_user_text(prompt_request: Any) -> str:
    """Extract concatenated user text from prompt_request.parts.

    Handles both dict parts ({"type": "text", "text": "..."}) and
    object parts (part.type == "text" and part.text).
    Returns empty string if no text found.
    """
    user_text = ""
    for part in prompt_request.parts:
        if isinstance(part, dict):
            if part.get("type") == "text":
                text_val = part.get("text", "")
                if text_val:
                    user_text += text_val
        elif hasattr(part, "type") and part.type == "text" and hasattr(part, "text"):
            text_val = part.text
            if text_val:
                user_text += text_val
    return user_text


def extract_provider_model(prompt_request: Any) -> tuple[str | None, str | None]:
    """Extract provider_id and model_id from prompt_request.

    Checks prompt_request.providerID/modelID first, then falls back to
    prompt_request.model.providerID/modelID.
    """
    provider_id = getattr(prompt_request, "providerID", None)
    model_id = getattr(prompt_request, "modelID", None)
    if not provider_id or not model_id:
        model_obj = getattr(prompt_request, "model", None)
        if model_obj:
            provider_id = getattr(model_obj, "providerID", None)
            model_id = getattr(model_obj, "modelID", None)
    return provider_id, model_id


def resolve_llm_config(
    provider_id: str,
    model_id: str,
    slog: Any,
) -> "LLMConfig | None":
    """Build LLMConfig from TUI-selected provider/model via provider registry.

    Looks up provider in _build_providers(), extracts api_key from env,
    returns LLMConfig or None if not found.
    """
    import os

    for provider in _build_providers():
        if provider.id != provider_id:
            continue
        model = provider.models.get(model_id)
        if not model:
            slog.warning("model_not_found", provider=provider_id, model=model_id)
            return None

        api_key = ""

        if provider.env:
            api_key = os.environ.get(provider.env[0], "")

        if not api_key and provider_id != "ollama":
            slog.warning("api_key_missing", provider=provider_id, env=provider.env)

        return LLMConfig(
            provider=provider_id,
            model=model.id,
            api_key=api_key or "not-set",
            api_base=model.api.url,
        )

    slog.warning("provider_not_found", provider=provider_id)
    return None
