"""
Independent pipeline phases (assignment: each phase runnable; Phase 6 = corrected only).

Artifacts (per variant: baseline | corrected), under output/<model_slug>/:
  model_slug from GENERATION_MODEL + JUDGE_MODEL (see paths.artifact_dir_slug).
  Phase 1: {variant}_phase1_raw.jsonl, {variant}_run_meta.json
  Phase 2: {variant}_dataset.jsonl, {variant}_invalid.jsonl (if any)
  Phase 3–4: {variant}_judgments.jsonl  (judge = failure modes + quality in one call)
  Phase 5:  {variant}_summary.json, generation_log_{variant}.jsonl, visualizations/*.png

Phase 6 (conceptual): run phases 1–5 for variant `corrected` after editing CORRECTED_TEMPLATES —
  does not read or overwrite baseline_* files.

Phase 7: benchmark (see run_benchmark_calibration in pipeline.py).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis import FAILURE_MODE_KEYS, aggregate_run, failure_cooccurrence_matrix
from config import Settings
from generator import generate_balanced_batch, structural_validate_items
from io_utils import read_jsonl, write_json, write_jsonl
from judge import evaluate_batch
from paths import artifact_dir_slug, output_root_for_settings, viz_dir_for_settings
from prompts import PromptVariant
from schemas import JudgeRecord, RunSummary
from visualizations import (
    ensure_viz_dir,
    plot_failure_by_category,
    plot_failure_cooccurrence,
    plot_most_problematic_count,
    plot_quality_dimensions_single_run,
)

logger = logging.getLogger(__name__)


def judgments_to_jsonl_rows(records: list[JudgeRecord | None]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in records:
        if r is None:
            continue
        d = r.model_dump()
        d["quality_scores"] = r.quality_scores.model_dump()
        out.append(d)
    return out


def generation_log_rows(
    settings: Settings,
    valid_rows: list[dict[str, Any]],
    records: list[JudgeRecord | None],
) -> list[dict[str, Any]]:
    by_trace: dict[str, JudgeRecord] = {
        r.trace_id: r for r in records if r is not None
    }
    out: list[dict[str, Any]] = []
    for row in valid_rows:
        tid = row.get("trace_id", "")
        rec = by_trace.get(tid)
        modes = (
            [k for k in FAILURE_MODE_KEYS if getattr(rec, k) == 1]
            if rec
            else []
        )
        out.append(
            {
                "trace_id": tid,
                "template_id": row.get("template_id"),
                "category": row.get("category"),
                "structural_valid": True,
                "failure_modes_flagged": modes,
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "model_used": settings.generation_model,
                "judge_model": settings.judge_model,
            }
        )
    return out


def _meta_path(variant: str, settings: Settings) -> Path:
    return output_root_for_settings(settings) / f"{variant}_run_meta.json"


def _read_meta(variant: str, settings: Settings) -> dict[str, Any]:
    p = _meta_path(variant, settings)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _write_meta(variant: str, data: dict[str, Any], settings: Settings) -> None:
    root = output_root_for_settings(settings)
    root.mkdir(parents=True, exist_ok=True)
    prev = _read_meta(variant, settings)
    prev.update(data)
    with _meta_path(variant, settings).open("w", encoding="utf-8") as f:
        json.dump(prev, f, indent=2, ensure_ascii=False)


def phase1_generation(
    settings: Settings, variant: PromptVariant, n: int
) -> Path:
    """Phase 1: diverse templates → LLM → structured rows (pre second-pass validation)."""
    rows, _logs = generate_balanced_batch(settings, n, variant)
    root = output_root_for_settings(settings)
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / f"{variant}_phase1_raw.jsonl"
    write_jsonl(raw_path, rows)
    _write_meta(
        variant,
        {
            "variant": variant,
            "artifact_subdir": artifact_dir_slug(settings),
            "num_samples_requested": n,
            "llm_success_count": len(rows),
        },
        settings,
    )
    logger.info("Phase 1 done: %s rows → %s", len(rows), raw_path.name)
    return raw_path


def phase2_structural_validation(
    settings: Settings, variant: PromptVariant
) -> tuple[Path, Path | None]:
    """Phase 2: Pydantic validate Phase 1 output → dataset + invalid."""
    root = output_root_for_settings(settings)
    raw_path = root / f"{variant}_phase1_raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}; run phase 1 first for variant={variant}"
        )
    rows = read_jsonl(raw_path)
    valid, invalid = structural_validate_items(rows)
    ds_path = root / f"{variant}_dataset.jsonl"
    write_jsonl(ds_path, valid)
    inv_path: Path | None = root / f"{variant}_invalid.jsonl"
    if invalid:
        write_jsonl(inv_path, invalid)
    else:
        if inv_path.exists():
            inv_path.unlink()
        inv_path = None
    _write_meta(
        variant,
        {
            "structural_valid_count": len(valid),
            "structural_invalid_count": len(invalid),
        },
        settings,
    )
    logger.info(
        "Phase 2 done: valid=%s invalid=%s → %s",
        len(valid),
        len(invalid),
        ds_path.name,
    )
    return ds_path, inv_path


def phase3_4_judge(settings: Settings, variant: PromptVariant) -> Path:
    """Phases 3–4: LLM-as-judge (failure modes + quality dimensions)."""
    root = output_root_for_settings(settings)
    ds_path = root / f"{variant}_dataset.jsonl"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"Missing {ds_path}; run phase 2 first for variant={variant}"
        )
    valid = read_jsonl(ds_path)
    records = evaluate_batch(settings, valid)
    j_path = root / f"{variant}_judgments.jsonl"
    write_jsonl(j_path, judgments_to_jsonl_rows(records))
    _write_meta(
        variant, {"judged_row_count": len([r for r in records if r])}, settings
    )
    logger.info("Phases 3–4 done: judgments → %s", j_path.name)
    return j_path


def _records_from_disk(
    valid_rows: list[dict[str, Any]],
    variant: PromptVariant,
    settings: Settings,
) -> list[JudgeRecord | None]:
    jpath = output_root_for_settings(settings) / f"{variant}_judgments.jsonl"
    if not jpath.exists():
        raise FileNotFoundError(f"Missing {jpath}; run phase 3 first")
    jrows = read_jsonl(jpath)
    by_id: dict[str, JudgeRecord] = {}
    for jr in jrows:
        by_id[jr["trace_id"]] = JudgeRecord.model_validate(jr)
    return [by_id.get(r.get("trace_id", "")) for r in valid_rows]


def phase5_analysis_and_plots(settings: Settings, variant: PromptVariant) -> RunSummary:
    """Phase 5: aggregate metrics + PNGs + summary JSON + generation log."""
    root = output_root_for_settings(settings)
    vd = viz_dir_for_settings(settings)
    ds_path = root / f"{variant}_dataset.jsonl"
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing {ds_path}; run phase 2 first")
    valid = read_jsonl(ds_path)
    records = _records_from_disk(valid, variant, settings)
    meta = _read_meta(variant, settings)
    num_requested = int(meta.get("num_samples_requested", len(valid)))
    llm_ok = int(meta.get("llm_success_count", len(valid)))
    categories_by_trace = {r["trace_id"]: r.get("category", "unknown") for r in valid}

    summary = aggregate_run(
        run_name=variant,
        num_generated_raw=num_requested,
        llm_success_count=llm_ok,
        structural_valid_count=len(valid),
        records=records,
        categories_by_trace=categories_by_trace,
    )
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / f"{variant}_summary.json", summary.model_dump())
    write_jsonl(
        root / f"generation_log_{variant}.jsonl",
        generation_log_rows(settings, valid, records),
    )

    ensure_viz_dir(vd)
    valid_recs = [r for r in records if r is not None]
    if valid_recs:
        labels, mat = failure_cooccurrence_matrix(valid_recs)
        plot_failure_cooccurrence(labels, mat, variant, viz_dir=vd)
        plot_failure_by_category(summary.category_failure_rates, variant, viz_dir=vd)
        plot_most_problematic_count(valid_recs, variant, viz_dir=vd)
        plot_quality_dimensions_single_run(
            summary.per_dimension_quality_pass_rates, variant, viz_dir=vd
        )
    logger.info("Phase 5 done: summary → %s_summary.json", variant)
    return summary


def run_phases_1_through_5(
    settings: Settings, variant: PromptVariant, n: int
) -> tuple[RunSummary, list[dict[str, Any]], list[JudgeRecord | None]]:
    """Phases 1–5 for one variant (same net effect as previous monolithic run_variant)."""
    phase1_generation(settings, variant, n)
    phase2_structural_validation(settings, variant)
    phase3_4_judge(settings, variant)
    summary = phase5_analysis_and_plots(settings, variant)
    valid = read_jsonl(output_root_for_settings(settings) / f"{variant}_dataset.jsonl")
    records = _records_from_disk(valid, variant, settings)
    return summary, valid, records


def run_phase6_corrected_only(settings: Settings, n: int) -> None:
    """
    Phase 6 (assignment): re-run phases 1–5 for `corrected` only (baseline artifacts unchanged).
    """
    logger.info(
        "Phase 6: phases 1–5 for variant=corrected only (baseline files not modified)"
    )
    run_phases_1_through_5(settings, "corrected", n)


# Aliases (case-insensitive) → phase number. Numeric strings "1"…"7" also work.
PHASE_ALIASES: dict[str, int] = {
    "1": 1,
    "generation": 1,
    "gen": 1,
    "generate": 1,
    "2": 2,
    "structural_validation": 2,
    "validation": 2,
    "3": 3,
    "judge": 3,
    "labeling": 3,
    "evaluation": 3,
    "5": 5,
    "analysis": 5,
    "analyze": 5,
    "aggregates": 5,
    "6": 6,
    "corrected_rerun": 6,
    "corrected_iteration": 6,
    "prompt_correction": 6,
    "7": 7,
    "benchmark_calibration": 7,
    "calibration": 7,
    "benchmark": 7,
}


def resolve_phase_identifier(token: str) -> int:
    """Return phase id 1, 2, 3, 5, 6, or 7 from a name, alias, or digit string."""
    key = token.strip().lower()
    if key not in PHASE_ALIASES:
        raise ValueError(
            f"Unknown phase {token!r}. Use 1,2,3,5,6,7 or a name such as: "
            "generation, structural_validation, judge, analysis, corrected_rerun, benchmark_calibration"
        )
    return PHASE_ALIASES[key]


PHASE_HELP = """
Phase map (use --phase <number> or any alias):

  generation (aliases: gen, generate)     Phase 1  → {variant}_phase1_raw.jsonl
  structural_validation (validation)      Phase 2  → {variant}_dataset.jsonl
  judge (labeling, evaluation)            Phase 3  → {variant}_judgments.jsonl  [brief: Phases 3–4]
  analysis (analyze, aggregates)          Phase 5  → summary + charts
  corrected_rerun (corrected_iteration, prompt_correction)  Phase 6  → full 1–5 for corrected only
  benchmark_calibration (calibration, benchmark)              Phase 7  → HF benchmark judge

Examples:
  python main.py run-phase --phase generation --variant baseline --num-samples 50
  python main.py run-phase --phase structural_validation --variant baseline
  python main.py run-phase --phase judge --variant baseline
  python main.py run-phase --phase analysis --variant baseline
  python main.py run-phase --phase corrected_rerun --num-samples 50
  python main.py run-phase --phase benchmark_calibration
"""
