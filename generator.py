"""Phase 1: LLM generation with Instructor + Pydantic validation + retries."""

from __future__ import annotations

import logging
import random
import time
import uuid
from typing import Any

from config import Settings
from llm_client import get_instructor_client
from prompts import PromptVariant, get_template
from schemas import DIYRepairItem, GenerationLogEntry, RepairCategory

logger = logging.getLogger(__name__)


def _looks_like_auth_failure(error: Exception) -> bool:
    text = str(error).lower()
    return "401" in text and (
        "unauthorized" in text
        or "authentication" in text
        or "user not found" in text
    )


def _user_message_for_generation(category: RepairCategory) -> str:
    return (
        "Generate exactly one DIY repair record. Fill every field. "
        "steps: JSON array of at least 3 plain strings. "
        "tips: JSON array of strings only — each element is one short sentence, "
        "no nested brackets or quotes inside a tip string (avoid starting a tip with '['). "
        "tools_required: array of strings."
    )


def generate_one(
    settings: Settings,
    category: RepairCategory,
    variant: PromptVariant,
    model: str | None = None,
) -> tuple[DIYRepairItem | None, GenerationLogEntry]:
    """Returns validated item or None with log entry."""
    client = get_instructor_client(settings)
    model = model or settings.generation_model
    template_body = get_template(category, variant)
    template_id = f"{variant}_{category}"
    trace_id = f"qa_{uuid.uuid4().hex[:12]}"

    system_prompt = template_body
    user_prompt = _user_message_for_generation(category)

    last_err: str | None = None
    for attempt in range(settings.max_retries):
        try:
            item: DIYRepairItem = client.chat.completions.create(
                model=model,
                temperature=settings.generation_temperature,
                response_model=DIYRepairItem,
                max_retries=1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            log = GenerationLogEntry(
                trace_id=trace_id,
                template_id=template_id,
                category=category,
                structural_valid=True,
                structural_errors=[],
                model_used=model,
            )
            return item, log
        except Exception as e:
            if _looks_like_auth_failure(e):
                raise RuntimeError(
                    "OpenRouter rejected OPENROUTER_API_KEY with 401. "
                    "Update .env with a valid raw sk-or-v1-... key, then rerun "
                    "`python3 scripts/openrouter_connectivity_check.py`."
                ) from e
            last_err = str(e)
            logger.warning("Generation attempt %s failed: %s", attempt + 1, e)
            time.sleep(settings.request_delay_seconds * (attempt + 1))

    log = GenerationLogEntry(
        trace_id=trace_id,
        template_id=template_id,
        category=category,
        structural_valid=False,
        structural_errors=[last_err or "unknown error"],
        model_used=model,
    )
    return None, log


def generate_balanced_batch(
    settings: Settings,
    n: int,
    variant: PromptVariant,
) -> tuple[list[dict[str, Any]], list[GenerationLogEntry]]:
    """
    Random category per item so that over many samples all five appear
    (shuffle categories to spread draws).
    """
    cats: list[RepairCategory] = [
        "appliance_repair",
        "plumbing_repair",
        "electrical_repair",
        "hvac_maintenance",
        "general_home_repair",
    ]
    out: list[dict[str, Any]] = []
    logs: list[GenerationLogEntry] = []
    for i in range(n):
        category = random.choice(cats)
        item, log = generate_one(settings, category, variant)
        time.sleep(settings.request_delay_seconds)
        if item is not None:
            row = item.model_dump()
            row["trace_id"] = log.trace_id
            row["template_id"] = log.template_id
            out.append(row)
        logs.append(log)
    return out, logs


def structural_validate_items(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Phase 2: split into valid / invalid by Pydantic."""
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for row in rows:
        try:
            DIYRepairItem.model_validate(row)
            valid.append(row)
        except Exception as e:
            row = dict(row)
            row["_validation_error"] = str(e)
            invalid.append(row)
    return valid, invalid
