#!/usr/bin/env python3
"""
Home DIY Repair Q&A synthetic data pipeline (mini-project-DIY).

Commands:
  baseline   — Phases 1–5 with baseline prompts (JSON/JSONL + charts)
  corrected  — Same with corrected prompts
  run-phase  — Run a single phase; --phase <number or name>, e.g. generation, structural_validation, judge, analysis (see README)
  validate   — Phase 2 only: Pydantic check on a JSONL file (no LLM / no API key)
  benchmark  — Phase 7: judge calibration on HF benchmark sample
  full       — baseline + corrected + benchmark + final report + comparison charts
  report     — Recompute final_report.json from saved summaries (no API calls)

Environment:
  LLM_PROVIDER         auto | groq | openrouter | openai (default: auto)
  GROQ_API_KEY         Groq (first in auto if set)
  OPENROUTER_API_KEY   OpenRouter (second in auto; base URL https://openrouter.ai/api/v1)
  OPENAI_API_KEY       OpenAI (third in auto)
  GENERATION_MODEL     (optional; use provider-specific ids, e.g. meta-llama/llama-3.1-8b-instruct on OpenRouter)
  JUDGE_MODEL          (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from analysis import most_problematic
from config import get_settings
from paths import output_root_for_settings, viz_dir_for_settings
from generator import structural_validate_items
from io_utils import read_jsonl, write_jsonl
from phases import (
    PHASE_HELP,
    phase1_generation,
    phase2_structural_validation,
    phase3_4_judge,
    phase5_analysis_and_plots,
    resolve_phase_identifier,
    run_phase6_corrected_only,
)
from pipeline import (
    build_final_report,
    compare_visualizations,
    run_benchmark_calibration,
    run_full_pipeline,
    run_variant,
)
from prompts import PromptVariant
from schemas import RunSummary

_PROJECT_DIR = Path(__file__).resolve().parent


def _default_baseline_dataset_path() -> Path | None:
    """Prefer legacy flat output, else newest output/<model_slug>/baseline_dataset.jsonl."""
    flat = _PROJECT_DIR / "output" / "baseline_dataset.jsonl"
    if flat.exists():
        return flat
    out = _PROJECT_DIR / "output"
    if not out.is_dir():
        return None
    candidates = [p for p in out.glob("*/baseline_dataset.jsonl") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def cmd_validate(input_path: Path | None) -> None:
    """Phase 2: re-run Pydantic validation on a JSONL (e.g. after manual edits)."""
    inp = input_path or _default_baseline_dataset_path()
    if inp is None or not inp.exists():
        print(
            "Input not found. Pass --input path/to.jsonl or place baseline_dataset.jsonl "
            "under output/ or output/<model_slug>/",
            file=sys.stderr,
        )
        sys.exit(1)
    rows = read_jsonl(inp)
    valid, invalid = structural_validate_items(rows)
    out_root = inp.parent
    out_root.mkdir(parents=True, exist_ok=True)
    stem = inp.stem
    out_valid = out_root / f"{stem}_validated.jsonl"
    out_invalid = out_root / f"{stem}_structural_invalid.jsonl"
    write_jsonl(out_valid, valid)
    write_jsonl(out_invalid, invalid)
    n = len(rows)
    rate = len(valid) / n if n else 0.0
    summary = {
        "input": str(inp.resolve()),
        "total_rows": n,
        "structural_valid_count": len(valid),
        "structural_invalid_count": len(invalid),
        "structural_pass_rate": round(rate, 4),
        "valid_output": str(out_valid.resolve()),
        "invalid_output": str(out_invalid.resolve()),
    }
    print(json.dumps(summary, indent=2))


def _load_summary(name: str, out_root: Path) -> RunSummary | None:
    p = out_root / f"{name}_summary.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return RunSummary.model_validate(json.load(f))


def cmd_report() -> None:
    settings = get_settings()
    out_root = output_root_for_settings(settings)
    b = _load_summary("baseline", out_root)
    c = _load_summary("corrected", out_root)
    bench = _load_summary("benchmark_calibration", out_root)
    if not b or not c:
        print(
            f"Need {out_root / 'baseline_summary.json'} and {out_root / 'corrected_summary.json'}",
            file=sys.stderr,
        )
        sys.exit(1)
    report = build_final_report(b, c, bench)
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "final_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    compare_visualizations(b, c, bench, viz_dir=viz_dir_for_settings(settings))
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="DIY synthetic Q&A pipeline")
    parser.add_argument(
        "command",
        choices=[
            "baseline",
            "corrected",
            "run-phase",
            "validate",
            "benchmark",
            "full",
            "report",
        ],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Items to generate per run (default: from env or 50)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="For 'validate' only: path to JSONL to check (default: output/<model_slug>/baseline_dataset.jsonl)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        metavar="NAME_OR_NUMBER",
        help="For 'run-phase': phase number (1,2,3,5,6,7) or name: generation, structural_validation, judge, analysis, corrected_rerun, benchmark_calibration (see phases.PHASE_ALIASES)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["baseline", "corrected"],
        default=None,
        help="For 'run-phase' phases 1–5: which artifact set (baseline or corrected)",
    )
    args = parser.parse_args()
    if args.command == "report":
        cmd_report()
        return
    if args.command == "validate":
        cmd_validate(args.input)
        return

    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info(
        "LLM provider=%s base_url=%s generation_model=%s judge_model=%s",
        settings.resolved_provider_label,
        settings.resolved_base_url,
        settings.generation_model,
        settings.judge_model,
    )
    n = args.num_samples or settings.default_num_samples
    if n < 50:
        logging.warning(
            "Course checklist recommends >= 50 samples per run; got %s", n
        )

    if args.command == "run-phase":
        if args.phase is None:
            print(PHASE_HELP, file=sys.stderr)
            sys.exit(1)
        try:
            pnum = resolve_phase_identifier(args.phase)
        except ValueError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        log.info("run-phase %r → phase %s", args.phase, pnum)
        if pnum == 7:
            s, _ = run_benchmark_calibration(settings, n=min(50, n))
            print(s.model_dump_json(indent=2))
            return
        if pnum == 6:
            run_phase6_corrected_only(settings, n)
            print(
                "Phase 6 done: corrected artifacts updated; baseline_* unchanged.",
                file=sys.stderr,
            )
            return
        if args.variant is None:
            print(
                "run-phase for phases 1–5 requires --variant baseline|corrected",
                file=sys.stderr,
            )
            sys.exit(1)
        v: PromptVariant = args.variant  # type: ignore[assignment]
        if pnum == 1:
            phase1_generation(settings, v, n)
            return
        if pnum == 2:
            phase2_structural_validation(settings, v)
            return
        if pnum == 3:
            phase3_4_judge(settings, v)
            return
        if pnum == 5:
            s = phase5_analysis_and_plots(settings, v)
            print(s.model_dump_json(indent=2))
            return
        return

    if args.command == "full":
        run_full_pipeline(settings, n)
        return
    if args.command == "benchmark":
        s, _ = run_benchmark_calibration(settings, n=min(50, n))
        print(s.model_dump_json(indent=2))
        return
    if args.command == "baseline":
        b_sum, b_valid, b_recs = run_variant(settings, "baseline", n)
        print("\n=== BASELINE SUMMARY ===\n")
        print(b_sum.model_dump_json(indent=2))
        worst = most_problematic(
            [r for r in b_recs if r is not None], b_valid, min_flags=3
        )
        print("\nExamples with 3+ failure modes (up to 5):", len(worst))
        for w in worst[:5]:
            print(w.get("trace_id"), w.get("_failure_modes"))
        return
    if args.command == "corrected":
        c_sum, c_valid, c_recs = run_variant(settings, "corrected", n)
        print("\n=== CORRECTED SUMMARY ===\n")
        print(c_sum.model_dump_json(indent=2))
        worst = most_problematic(
            [r for r in c_recs if r is not None], c_valid, min_flags=3
        )
        print("\nExamples with 3+ failure modes (up to 5):", len(worst))
        for w in worst[:5]:
            print(w.get("trace_id"), w.get("_failure_modes"))
        return


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
