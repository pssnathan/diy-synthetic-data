#!/usr/bin/env python3
"""
One-shot OpenRouter connectivity check (OpenAI-compatible Chat Completions).

Uses OPENROUTER_API_KEY, OPENROUTER_BASE_URL, GENERATION_MODEL from mini-project-DIY/.env.
Optional: OPENROUTER_HTTP_REFERER, OPENROUTER_APP_NAME (recommended by OpenRouter).

Run:

  cd mini-project-DIY && python scripts/openrouter_connectivity_check.py

OpenRouter base URL defaults to https://openrouter.ai/api/v1 .
Model ids often contain slashes (e.g. meta-llama/llama-3.1-8b-instruct); only leading/trailing
whitespace is stripped — do not rewrite the slug.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openai import OpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenRouterSmokeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = Field(validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL",
    )
    generation_model: str = Field(
        default="meta-llama/llama-3.1-8b-instruct",
        validation_alias="GENERATION_MODEL",
    )
    openrouter_http_referer: str | None = Field(
        default=None, validation_alias="OPENROUTER_HTTP_REFERER"
    )
    openrouter_app_name: str | None = Field(
        default=None, validation_alias="OPENROUTER_APP_NAME"
    )


def main() -> int:
    try:
        s = OpenRouterSmokeSettings()
    except Exception as e:
        print("Failed to load settings from .env:", e, file=sys.stderr)
        print(
            "Ensure OPENROUTER_API_KEY is set in mini-project-DIY/.env",
            file=sys.stderr,
        )
        return 1

    model = s.generation_model.strip()
    base_url = s.openrouter_base_url.rstrip("/")
    headers: dict[str, str] = {}
    ref = (s.openrouter_http_referer or "").strip()
    if ref:
        headers["HTTP-Referer"] = ref
    title = (s.openrouter_app_name or "").strip()
    if title:
        headers["X-OpenRouter-Title"] = title

    kwargs: dict = {
        "api_key": s.openrouter_api_key.strip(),
        "base_url": base_url,
    }
    if headers:
        kwargs["default_headers"] = headers

    client = OpenAI(**kwargs)

    print(f"Base URL: {base_url}")
    print(f"Model: {model!r}")
    if headers:
        print("Extra headers:", list(headers.keys()))

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a terse assistant. Reply in one short sentence.",
                },
                {"role": "user", "content": 'Say exactly: "pong" if you can read this.'},
            ],
            max_tokens=64,
            temperature=0,
        )
    except Exception as e:
        print("API call failed:", e, file=sys.stderr)
        if getattr(e, "status_code", None) == 401:
            print(
                "\nOpenRouter rejected OPENROUTER_API_KEY. Create a new key at "
                "https://openrouter.ai/settings/keys and paste only the raw "
                "sk-or-v1-... value into .env; do not include 'Bearer '.",
                file=sys.stderr,
            )
        else:
            print(
                "\nUse the exact model id from https://openrouter.ai/models "
                "(e.g. meta-llama/llama-3.1-8b-instruct).",
                file=sys.stderr,
            )
        return 2

    choice = resp.choices[0].message
    text = (choice.content or "").strip()
    print("Response role:", choice.role)
    print("Response content:", text[:500] + ("…" if len(text) > 500 else ""))
    print("OK: OpenRouter connectivity and model id accepted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
