"""Phases 3–4: LLM-as-Judge with Instructor; deterministic temperature."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from category_keywords import keyword_matches_category
from config import Settings
from llm_client import get_instructor_client
from schemas import DIYRepairItem, JudgeOutputRaw, JudgeRecord, QualityScores

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """You are an expert evaluator for home DIY repair training data.
You must score ONE item at a time.

Failure modes (binary): 0 = pass, 1 = fail
- incomplete_answer: not enough detail to complete the repair safely and practically
- safety_violations: missing or wrong safety guidance for hazards present
- unrealistic_tools: needs professional/specialty tools or unrealistic for a homeowner
- overcomplicated_solution: tells user to hire pros for a simple DIY task incorrectly
- missing_context: question/answer lacks context to understand the problem
- poor_quality_tips: tips vague, generic, or unhelpful

Quality dimensions (binary): 1 = pass, 0 = fail
- answer_coherence: answer reads as one narrative, not a stitched list
- step_actionability: steps are specific (no empty "properly"/"as needed" without detail)
- tool_realism: tools are homeowner-realistic and affordable
- safety_specificity: names specific hazard and precaution (not just "be careful")
- tip_usefulness: tips add non-obvious value beyond steps
- problem_answer_alignment: answer fixes the stated equipment_problem
- appropriate_scope: DIY-appropriate; defers to pros when needed
- category_accuracy: category matches the actual domain of the content

Be calibrated: high-quality reference data should PASS almost all dimensions.
Only fail when there is a clear defect. When unsure, prefer PASS for benchmark-like quality."""

SAFETY_MIN_CHARS = 80


def _apply_rule_overrides(
    item: DIYRepairItem, raw: JudgeOutputRaw
) -> QualityScores:
    """Enforce Q4 length and Q8 keyword heuristics on top of LLM scores."""
    qs = raw.quality_scores.model_copy()
    if len(item.safety_info.strip()) < SAFETY_MIN_CHARS:
        qs.safety_specificity = 0
    if not keyword_matches_category(
        item.category, item.question, item.answer, item.equipment_problem
    ):
        qs.category_accuracy = 0
    return qs


def _derive_flags(raw: JudgeOutputRaw, qs: QualityScores) -> tuple[bool, bool]:
    modes = [
        raw.incomplete_answer,
        raw.safety_violations,
        raw.unrealistic_tools,
        raw.overcomplicated_solution,
        raw.missing_context,
        raw.poor_quality_tips,
    ]
    overall_failure = any(m == 1 for m in modes)
    quality_pass = all(
        getattr(qs, name) == 1
        for name in (
            "answer_coherence",
            "step_actionability",
            "tool_realism",
            "safety_specificity",
            "tip_usefulness",
            "problem_answer_alignment",
            "appropriate_scope",
            "category_accuracy",
        )
    )
    return overall_failure, quality_pass


def evaluate_item(
    settings: Settings,
    item: DIYRepairItem,
    trace_id: str,
    model: str | None = None,
) -> JudgeRecord | None:
    client = get_instructor_client(settings)
    model = model or settings.judge_model
    payload = json.dumps(item.model_dump(), ensure_ascii=False)
    user = f"trace_id: {trace_id}\n\nEvaluate this JSON item:\n{payload}"

    for attempt in range(settings.max_retries):
        try:
            raw: JudgeOutputRaw = client.chat.completions.create(
                model=model,
                temperature=settings.judge_temperature,
                response_model=JudgeOutputRaw,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
            raw = raw.model_copy(update={"trace_id": trace_id})
            qs = _apply_rule_overrides(item, raw)
            overall_failure, quality_pass = _derive_flags(raw, qs)
            return JudgeRecord(
                trace_id=trace_id,
                incomplete_answer=raw.incomplete_answer,
                safety_violations=raw.safety_violations,
                unrealistic_tools=raw.unrealistic_tools,
                overcomplicated_solution=raw.overcomplicated_solution,
                missing_context=raw.missing_context,
                poor_quality_tips=raw.poor_quality_tips,
                overall_failure=overall_failure,
                quality_scores=qs,
                quality_pass=quality_pass,
            )
        except Exception as e:
            logger.warning("Judge attempt %s failed: %s", attempt + 1, e)
            time.sleep(settings.request_delay_seconds * (attempt + 1))
    return None


def dict_to_item(row: dict[str, Any]) -> DIYRepairItem:
    keys = {f for f in DIYRepairItem.model_fields}
    return DIYRepairItem.model_validate({k: v for k, v in row.items() if k in keys})


def evaluate_batch(
    settings: Settings,
    rows: list[dict[str, Any]],
) -> list[JudgeRecord | None]:
    results: list[JudgeRecord | None] = []
    for row in rows:
        trace_id = row.get("trace_id") or "unknown"
        try:
            item = dict_to_item(row)
        except Exception as e:
            logger.warning("Skip judge row %s: %s", trace_id, e)
            results.append(None)
            continue
        rec = evaluate_item(settings, item, trace_id)
        results.append(rec)
        time.sleep(settings.request_delay_seconds)
    return results
