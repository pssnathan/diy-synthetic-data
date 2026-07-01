"""Environment-backed settings for the DIY synthetic data pipeline."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Model ids that work on Groq’s OpenAI-compatible API but not on api.openai.com
_GROQ_HOSTED_MODEL_IDS: frozenset[str] = frozenset(
    {
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    }
)


def _normalized_model_id(name: str) -> str:
    return re.sub(r"\s+", "", name.strip().lower())


def _looks_like_groq_only_model(model: str) -> bool:
    m = _normalized_model_id(model)
    if m in _GROQ_HOSTED_MODEL_IDS:
        return True
    if "llama-3.3-70b" in m or "llama-3.1-8b-instant" in m:
        return True
    return False


class Settings(BaseSettings):
    """
    LLM access uses an OpenAI-compatible client (Instructor + OpenAI SDK).

    Provider selection:
      - LLM_PROVIDER=auto (default): Groq if GROQ_API_KEY set, else OpenRouter if
        OPENROUTER_API_KEY set, else OpenAI if OPENAI_API_KEY set.
      - LLM_PROVIDER=groq: require GROQ_API_KEY.
      - LLM_PROVIDER=openrouter: require OPENROUTER_API_KEY; base URL defaults to
        https://openrouter.ai/api/v1 (OpenAI-compatible).
      - LLM_PROVIDER=openai: require OPENAI_API_KEY.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["auto", "groq", "openai", "openrouter"] = Field(
        default="auto",
        validation_alias="LLM_PROVIDER",
    )
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openrouter_api_key: str | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        validation_alias="GROQ_BASE_URL",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="OPENAI_BASE_URL",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL",
    )
    # Optional; OpenRouter recommends these for analytics (https://openrouter.ai/docs)
    openrouter_http_referer: str | None = Field(
        default=None, validation_alias="OPENROUTER_HTTP_REFERER"
    )
    openrouter_app_name: str | None = Field(
        default=None, validation_alias="OPENROUTER_APP_NAME"
    )
    # Defaults favor Groq models when using Groq; set GENERATION_MODEL / JUDGE_MODEL in .env for OpenAI.
    generation_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias="GENERATION_MODEL",
    )
    judge_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias="JUDGE_MODEL",
    )
    generation_temperature: float = Field(default=0.75, ge=0.0, le=2.0)
    judge_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    request_delay_seconds: float = Field(default=0.35, ge=0.0)
    max_retries: int = Field(default=3, ge=1, le=10)
    default_num_samples: int = Field(default=50, ge=1)

    # Filled by validator — used by the OpenAI client
    resolved_api_key: str = ""
    resolved_base_url: str = ""
    resolved_provider_label: str = ""

    @staticmethod
    def _key(s: str | None) -> str | None:
        if s is None:
            return None
        t = str(s).strip()
        return t if t else None

    @model_validator(mode="after")
    def resolve_llm_credentials(self) -> Settings:
        gq = self._key(self.groq_api_key)
        oa = self._key(self.openai_api_key)
        or_key = self._key(self.openrouter_api_key)
        p = self.llm_provider
        if p == "auto":
            if gq:
                self.resolved_api_key = gq
                self.resolved_base_url = self.groq_base_url.rstrip("/")
                self.resolved_provider_label = "groq"
            elif or_key:
                self.resolved_api_key = or_key
                self.resolved_base_url = self.openrouter_base_url.rstrip("/")
                self.resolved_provider_label = "openrouter"
            elif oa:
                self.resolved_api_key = oa
                self.resolved_base_url = self.openai_base_url.rstrip("/")
                self.resolved_provider_label = "openai"
            else:
                raise ValueError(
                    "Set at least one of GROQ_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY "
                    "in .env (auto order: Groq → OpenRouter → OpenAI)."
                )
        elif p == "groq":
            if not gq:
                raise ValueError("LLM_PROVIDER=groq requires GROQ_API_KEY")
            self.resolved_api_key = gq
            self.resolved_base_url = self.groq_base_url.rstrip("/")
            self.resolved_provider_label = "groq"
        elif p == "openrouter":
            if not or_key:
                raise ValueError("LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY")
            self.resolved_api_key = or_key
            self.resolved_base_url = self.openrouter_base_url.rstrip("/")
            self.resolved_provider_label = "openrouter"
        else:  # openai
            if not oa:
                raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
            self.resolved_api_key = oa
            self.resolved_base_url = self.openai_base_url.rstrip("/")
            self.resolved_provider_label = "openai"
        self._reject_groq_models_on_openai_host()
        return self

    def _reject_groq_models_on_openai_host(self) -> None:
        """Groq model names against api.openai.com → 404 model_not_found; fail fast with a clear message."""
        if self.resolved_provider_label != "openai":
            return
        host = self.resolved_base_url.lower()
        if "api.openai.com" not in host and "openai.azure.com" not in host:
            return
        bad: list[str] = []
        if _looks_like_groq_only_model(self.generation_model):
            bad.append(f"GENERATION_MODEL={self.generation_model!r}")
        if _looks_like_groq_only_model(self.judge_model):
            bad.append(f"JUDGE_MODEL={self.judge_model!r}")
        if not bad:
            return
        raise ValueError(
            "OpenAI’s API does not host these Groq-specific model ids: "
            + ", ".join(bad)
            + ". Set GENERATION_MODEL and JUDGE_MODEL to OpenAI model ids "
            "(e.g. gpt-4o-mini, gpt-5.4-mini), or use LLM_PROVIDER=groq with GROQ_API_KEY "
            "and GROQ_BASE_URL for Llama on Groq."
        )


def get_settings() -> Settings:
    return Settings()
