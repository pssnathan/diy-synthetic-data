#!/usr/bin/env python3
"""
One-shot OpenAI connectivity check (Chat Completions).

Uses OPENAI_API_KEY, OPENAI_BASE_URL, and GENERATION_MODEL from mini-project-DIY/.env.
Run from repo root:

  cd mini-project-DIY && python scripts/openai_connectivity_check.py

Model names in .env may include spaces or mixed case; they are normalized to a
typical API id (e.g. "GPT-5.4 mini" -> "gpt-5.4-mini").
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


class OpenAISmokeSettings(BaseSettings):
    """Reads only OpenAI-related vars (does not require GROQ to be unset)."""

    model_config = SettingsConfigDict(
        env_file=str(_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="OPENAI_BASE_URL",
    )
    generation_model: str = Field(
        default="gpt-5.4-mini",
        validation_alias="GENERATION_MODEL",
    )


def normalize_model_id(raw: str) -> str:
    t = raw.strip().lower().replace(" ", "-")
    while "--" in t:
        t = t.replace("--", "-")
    return t


def main() -> int:
    try:
        s = OpenAISmokeSettings()
    except Exception as e:
        print("Failed to load settings from .env:", e, file=sys.stderr)
        print("Ensure OPENAI_API_KEY is set in mini-project-DIY/.env", file=sys.stderr)
        return 1

    model = normalize_model_id(s.generation_model)
    base_url = s.openai_base_url.rstrip("/")
    client = OpenAI(api_key=s.openai_api_key.strip(), base_url=base_url)

    print(f"Base URL: {base_url}")
    print(f"Model (raw -> normalized): {s.generation_model!r} -> {model!r}")

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
            max_completion_tokens=32,
            temperature=0,
        )
    except Exception as e:
        print("API call failed:", e, file=sys.stderr)
        print(
            "\nIf the model id was rejected, set GENERATION_MODEL to the exact id from "
            "https://platform.openai.com/docs/models (e.g. gpt-5.4-mini).",
            file=sys.stderr,
        )
        return 2

    choice = resp.choices[0].message
    text = (choice.content or "").strip()
    print("Response role:", choice.role)
    print("Response content:", text[:500] + ("…" if len(text) > 500 else ""))
    print("OK: connectivity and model id accepted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
