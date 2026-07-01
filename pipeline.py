"""Orchestration for baseline, corrected, benchmark calibration, and reporting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from analysis import QUALITY_KEYS, aggregate_run, improvement_ratio, most_problematic
from benchmark import load_benchmark_rows, validate_benchmark_compatibility
from config import Settings
from io_utils import write_json, write_jsonl
from judge import evaluate_batch
from paths import output_root_for_settings, viz_dir_for_settings
from phases import judgments_to_jsonl_rows, run_phases_1_through_5
from prompts import PromptVariant
from schemas import JudgeRecord, RunSummary
from visualizations import (
    plot_benchmark_vs_generated,
    plot_per_mode_trend,
    plot_quality_dimensions,
)

logger = logging.getLogger(__name__)


def run_variant(
    settings: Settings,
    variant: PromptVariant,
    n: int,
) -> tuple[RunSummary, list[dict[str, Any]], list[JudgeRecord | None]]:
    """Phases 1–5 for one prompt variant (delegates to phases.run_phases_1_through_5)."""
    return run_phases_1_through_5(settings, variant, n)


def run_benchmark_calibration(
    settings: Settings,
    n: int = 50,
) -> tuple[RunSummary, list[JudgeRecord | None]]:
    """Phase 7: judge on benchmark sample."""
    rows = load_benchmark_rows(n=n)
    rows = validate_benchmark_compatibility(rows)
    if len(rows) < n:
        logger.warning("Only %s benchmark rows passed schema check", len(rows))
    records = evaluate_batch(settings, rows)
    categories_by_trace = {r["trace_id"]: r.get("category", "unknown") for r in rows}
    summary = aggregate_run(
        run_name="benchmark_calibration",
        num_generated_raw=len(rows),
        llm_success_count=len(rows),
        structural_valid_count=len(rows),
        records=records,
        categories_by_trace=categories_by_trace,
    )
    out = output_root_for_settings(settings)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "benchmark_calibration_summary.json", summary.model_dump())
    write_jsonl(
        out / "benchmark_calibration_judgments.jsonl",
        judgments_to_jsonl_rows(records),
    )
    return summary, records


def build_final_report(
    baseline: RunSummary,
    corrected: RunSummary,
    benchmark: RunSummary | None,
) -> dict[str, Any]:
    imp = improvement_ratio(
        baseline.overall_failure_rate, corrected.overall_failure_rate
    )
    report: dict[str, Any] = {
        "baseline_failure_rate": baseline.overall_failure_rate,
        "corrected_failure_rate": corrected.overall_failure_rate,
        "improvement_ratio": imp,
        "improvement_target_met": imp >= 0.80,
        "baseline_quality_pass_rate": baseline.quality_pass_rate,
        "corrected_quality_pass_rate": corrected.quality_pass_rate,
        "per_mode_baseline": baseline.per_mode_failure_rates,
        "per_mode_corrected": corrected.per_mode_failure_rates,
        "per_dimension_baseline": baseline.per_dimension_quality_pass_rates,
        "per_dimension_corrected": corrected.per_dimension_quality_pass_rates,
    }
    if benchmark:
        worst_dim = None
        min_rate = 1.0
        for q in QUALITY_KEYS:
            r = benchmark.per_dimension_quality_pass_rates.get(q, 1.0)
            if r < min_rate:
                min_rate = r
                worst_dim = q
        report["benchmark_worst_dimension_pass_rate"] = min_rate
        report["benchmark_worst_dimension"] = worst_dim
        report["benchmark_calibration_threshold_met"] = min_rate >= 0.95
        report["benchmark_per_dimension"] = benchmark.per_dimension_quality_pass_rates
        report["quality_gap_corrected_minus_benchmark"] = {
            q: corrected.per_dimension_quality_pass_rates.get(q, 0.0)
            - benchmark.per_dimension_quality_pass_rates.get(q, 0.0)
            for q in QUALITY_KEYS
        }
    return report


def compare_visualizations(
    baseline: RunSummary,
    corrected: RunSummary,
    benchmark: RunSummary | None,
    viz_dir: Path | None = None,
) -> None:
    plot_per_mode_trend(
        baseline.per_mode_failure_rates,
        corrected.per_mode_failure_rates,
        viz_dir=viz_dir,
    )
    plot_quality_dimensions(
        baseline.per_dimension_quality_pass_rates,
        corrected.per_dimension_quality_pass_rates,
        viz_dir=viz_dir,
    )
    if benchmark:
        plot_benchmark_vs_generated(
            benchmark.per_dimension_quality_pass_rates,
            corrected.per_dimension_quality_pass_rates,
            viz_dir=viz_dir,
        )


def print_text_report(
    baseline: RunSummary,
    corrected: RunSummary,
    benchmark: RunSummary | None,
    valid_rows_baseline: list[dict[str, Any]],
    records_baseline: list[JudgeRecord | None],
) -> None:
    """Console summary + failure patterns."""
    print("\n=== BASELINE ===")
    print(baseline.model_dump_json(indent=2))
    print("\n=== CORRECTED ===")
    print(corrected.model_dump_json(indent=2))
    if benchmark:
        print("\n=== BENCHMARK CALIBRATION (judge on HF sample) ===")
        print(benchmark.model_dump_json(indent=2))

    rep = build_final_report(baseline, corrected, benchmark)
    print("\n=== IMPROVEMENT ===")
    print(
        f"improvement_ratio={rep['improvement_ratio']:.3f} "
        f"(target >= 0.80): {rep['improvement_target_met']}"
    )

    worst = most_problematic(
        [r for r in records_baseline if r is not None],
        valid_rows_baseline,
        min_flags=3,
    )
    print("\n=== EXAMPLE: items with 3+ failure modes (baseline) ===")
    for w in worst[:5]:
        print(w.get("trace_id"), w.get("_failure_modes"))


def run_full_pipeline(settings: Settings, n: int) -> None:
    logging.basicConfig(level=logging.INFO)
    b_sum, b_valid, b_recs = run_variant(settings, "baseline", n)
    c_sum, _, _ = run_variant(settings, "corrected", n)
    bench_sum, _ = run_benchmark_calibration(settings, n=min(50, n))

    vd = viz_dir_for_settings(settings)
    compare_visualizations(b_sum, c_sum, bench_sum, viz_dir=vd)
    report = build_final_report(b_sum, c_sum, bench_sum)
    out = output_root_for_settings(settings)
    write_json(out / "final_report.json", report)
    print_text_report(b_sum, c_sum, bench_sum, b_valid, b_recs)
    print(f"\nArtifacts written under {out} and {vd}")
