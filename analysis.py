"""Phase 5: aggregates, co-occurrence, category breakdown, worst items."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from schemas import JudgeRecord, RunSummary

FAILURE_MODE_KEYS: tuple[str, ...] = (
    "incomplete_answer",
    "safety_violations",
    "unrealistic_tools",
    "overcomplicated_solution",
    "missing_context",
    "poor_quality_tips",
)

QUALITY_KEYS: tuple[str, ...] = (
    "answer_coherence",
    "step_actionability",
    "tool_realism",
    "safety_specificity",
    "tip_usefulness",
    "problem_answer_alignment",
    "appropriate_scope",
    "category_accuracy",
)


def failure_cooccurrence_matrix(
    records: Sequence[JudgeRecord],
) -> tuple[list[str], list[list[float]]]:
    """Counts of joint failures (upper triangle + diagonal); normalized optional."""
    labels = list(FAILURE_MODE_KEYS)
    n = len(labels)
    mat = [[0.0 for _ in range(n)] for _ in range(n)]
    for r in records:
        flags = [getattr(r, k) for k in labels]
        for i in range(n):
            for j in range(n):
                if flags[i] == 1 and flags[j] == 1:
                    mat[i][j] += 1.0
    return labels, mat


def count_failure_flags_per_item(records: Sequence[JudgeRecord]) -> list[int]:
    out: list[int] = []
    for r in records:
        c = sum(1 for k in FAILURE_MODE_KEYS if getattr(r, k) == 1)
        out.append(c)
    return out


def aggregate_run(
    run_name: str,
    num_generated_raw: int,
    llm_success_count: int,
    structural_valid_count: int,
    records: list[JudgeRecord | None],
    categories_by_trace: dict[str, str],
) -> RunSummary:
    """categories_by_trace: trace_id -> repair category from generation."""
    valid_records = [r for r in records if r is not None]
    n = len(valid_records)
    struct_rate = structural_valid_count / max(llm_success_count, 1)
    if n == 0:
        return RunSummary(
            run_name=run_name,
            num_generated=num_generated_raw,
            structural_pass_count=structural_valid_count,
            structural_pass_rate=struct_rate,
            overall_failure_rate=0.0,
            per_mode_failure_rates={k: 0.0 for k in FAILURE_MODE_KEYS},
            quality_pass_rate=0.0,
            per_dimension_quality_pass_rates={k: 0.0 for k in QUALITY_KEYS},
            items_with_three_plus_failures=0,
            category_failure_rates={},
        )

    per_mode = {
        k: sum(1 for r in valid_records if getattr(r, k) == 1) / n for k in FAILURE_MODE_KEYS
    }
    q_pass = sum(1 for r in valid_records if r.quality_pass) / n
    per_dim = {}
    for qk in QUALITY_KEYS:
        per_dim[qk] = (
            sum(1 for r in valid_records if getattr(r.quality_scores, qk) == 1) / n
        )
    overall_fail = sum(1 for r in valid_records if r.overall_failure) / n
    three_plus = sum(
        1
        for r in valid_records
        if sum(1 for k in FAILURE_MODE_KEYS if getattr(r, k) == 1) >= 3
    )

    cat_fail: dict[str, list[bool]] = defaultdict(list)
    for r in valid_records:
        cat = categories_by_trace.get(r.trace_id, "unknown")
        cat_fail[cat].append(r.overall_failure)

    category_failure_rates = {
        c: sum(fs) / len(fs) for c, fs in cat_fail.items() if fs
    }

    return RunSummary(
        run_name=run_name,
        num_generated=num_generated_raw,
        structural_pass_count=structural_valid_count,
        structural_pass_rate=struct_rate,
        overall_failure_rate=overall_fail,
        per_mode_failure_rates=per_mode,
        quality_pass_rate=q_pass,
        per_dimension_quality_pass_rates=per_dim,
        items_with_three_plus_failures=three_plus,
        category_failure_rates=category_failure_rates,
    )


def improvement_ratio(baseline_rate: float, corrected_rate: float) -> float:
    if baseline_rate <= 0:
        return 0.0
    return (baseline_rate - corrected_rate) / baseline_rate


def most_problematic(
    records: Sequence[JudgeRecord], rows: list[dict[str, Any]], min_flags: int = 3
) -> list[dict[str, Any]]:
    """Join trace_id to row; return items with >= min_flags failure modes."""
    by_id = {r.trace_id: r for r in records}
    out: list[dict[str, Any]] = []
    for row in rows:
        tid = row.get("trace_id")
        r = by_id.get(tid)
        if not r:
            continue
        nfail = sum(1 for k in FAILURE_MODE_KEYS if getattr(r, k) == 1)
        if nfail >= min_flags:
            entry = dict(row)
            entry["_failure_flag_count"] = nfail
            entry["_failure_modes"] = [k for k in FAILURE_MODE_KEYS if getattr(r, k) == 1]
            out.append(entry)
    out.sort(key=lambda x: -x["_failure_flag_count"])
    return out


def summarize_for_report(summary: RunSummary) -> dict[str, Any]:
    return summary.model_dump()
