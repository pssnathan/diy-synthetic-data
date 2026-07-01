"""Resolved artifact directories (per LLM model pair) under output/."""

from __future__ import annotations

import re
from pathlib import Path

from config import Settings

_OUTPUT_BASE = Path(__file__).resolve().parent / "output"


def _sanitize_model_segment(name: str) -> str:
    """Filesystem-safe slug from a model id (Groq/OpenAI/etc.)."""
    t = name.strip().lower().replace(" ", "-")
    for ch in r'/\:*?"<>|':
        t = t.replace(ch, "_")
    t = re.sub(r"-+", "-", t).strip("-_") or "unknown-model"
    return t


def artifact_dir_slug(settings: Settings) -> str:
    """
    Subdirectory name under output/ for this run's artifacts.

    Uses generation + judge models so runs differ when only the judge changes.
    If both are identical, a single segment is used (shorter path).
    """
    g = _sanitize_model_segment(settings.generation_model)
    j = _sanitize_model_segment(settings.judge_model)
    if g == j:
        return g
    return f"{g}__{j}"


def output_root_for_settings(settings: Settings) -> Path:
    """Root directory for JSON/JSONL (e.g. output/gpt-5.4-mini/)."""
    return _OUTPUT_BASE / artifact_dir_slug(settings)


def viz_dir_for_settings(settings: Settings) -> Path:
    """PNG charts for this model pair (nested under that run's output/)."""
    return output_root_for_settings(settings) / "visualizations"
