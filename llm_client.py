"""OpenAI-compatible client patched with Instructor for structured outputs."""

from __future__ import annotations

import instructor
from openai import OpenAI

from config import Settings


def get_instructor_client(settings: Settings):
    kwargs: dict = {
        "api_key": settings.resolved_api_key,
        "base_url": settings.resolved_base_url,
    }
    if settings.resolved_provider_label == "openrouter":
        headers: dict[str, str] = {}
        ref = (settings.openrouter_http_referer or "").strip()
        if ref:
            headers["HTTP-Referer"] = ref
        title = (settings.openrouter_app_name or "").strip()
        if title:
            headers["X-OpenRouter-Title"] = title
        if headers:
            kwargs["default_headers"] = headers
    client = OpenAI(**kwargs)
    # Default Instructor TOOLS mode expects a single tool_call; many OpenRouter
    # models (e.g. Llama 8B) return JSON in content instead → parse failures and
    # wasted retries. OpenRouter structured mode uses response_format json_schema.
    if settings.resolved_provider_label == "openrouter":
        return instructor.from_openai(
            client, mode=instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS
        )
    return instructor.from_openai(client)
