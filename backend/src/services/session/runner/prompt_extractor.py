from typing import Any

from backend.src.api.schemas.provider_registry import _build_providers
from backend.src.settings import LLMConfig


def extract_user_text(prompt_request: Any) -> str:
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
    import os
    from pydantic import SecretStr

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

        # Use SecretStr to preserve case of API key (BaseModelFlexLower applies str_to_lower)
        # Use the original model_id to preserve case
        return LLMConfig(
            provider=provider_id,
            model=model_id,  # Use original model_id to preserve case
            api_key=SecretStr(api_key) if api_key else SecretStr("not-set"),
            api_base=model.api.url,
        )

    slog.warning("provider_not_found", provider=provider_id)
    return None
