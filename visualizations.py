"""Charts saved under visualizations/ (Matplotlib + Seaborn)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analysis import FAILURE_MODE_KEYS, QUALITY_KEYS
from schemas import JudgeRecord, RunSummary

VIZ_DIR = Path(__file__).resolve().parent / "visualizations"


def ensure_viz_dir(viz_dir: Path | None = None) -> Path:
    """Create directory for charts; default is top-level visualizations/."""
    root = viz_dir or VIZ_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def plot_failure_cooccurrence(
    labels: list[str],
    matrix: list[list[float]],
    name: str,
    viz_dir: Path | None = None,
) -> Path:
    root = ensure_viz_dir(viz_dir)
    arr = np.array(matrix)
    plt.figure(figsize=(10, 8))
    vmax = float(arr.max()) if arr.size else 0.0
    # All-zero matrix looks "blank"; force a sensible scale and note.
    ax = sns.heatmap(
        arr,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        square=True,
        vmin=0,
        vmax=max(vmax, 1.0),
    )
    plt.title("Failure mode co-occurrence (count of items failing both modes)")
    if vmax == 0:
        ax.text(
            0.5,
            0.5,
            "No failure-mode flags in this run\n(co-occurrence counts are all 0)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        )
    plt.tight_layout()
    path = root / f"failure_cooccurrence_{name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_failure_by_category(
    category_failure_rates: dict[str, float],
    name: str,
    viz_dir: Path | None = None,
) -> Path:
    root = ensure_viz_dir(viz_dir)
    cats = list(category_failure_rates.keys())
    vals = [category_failure_rates[c] for c in cats]
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette("viridis", n_colors=max(len(cats), 1))
    plt.bar(cats, vals, color=colors)
    plt.ylabel("Overall failure rate (any failure mode = fail)")
    plt.xlabel("Repair category (only categories present in this run)")
    plt.title("Failure rates by repair category")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1.0)
    if not cats:
        plt.text(0.5, 0.5, "No items to plot", ha="center", va="center", transform=plt.gca().transAxes)
    elif max(vals, default=0) == 0:
        plt.text(
            0.5,
            0.92,
            "All bars are 0: no overall_failure in judged items (good) — scale 0–1",
            ha="center",
            fontsize=10,
            transform=plt.gca().transAxes,
        )
    plt.tight_layout()
    path = root / f"failure_by_category_{name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_per_mode_trend(
    baseline: dict[str, float],
    corrected: dict[str, float],
    viz_dir: Path | None = None,
) -> Path:
    root = ensure_viz_dir(viz_dir)
    keys = list(FAILURE_MODE_KEYS)
    b = [baseline.get(k, 0) for k in keys]
    c = [corrected.get(k, 0) for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - w / 2, b, width=w, label="Baseline", color="#c44e52")
    plt.bar(x + w / 2, c, width=w, label="Corrected", color="#4c72b0")
    plt.xticks(x, keys, rotation=30, ha="right")
    plt.ylabel("Failure rate")
    plt.title("Per-mode failure rates: baseline vs corrected")
    plt.legend()
    plt.tight_layout()
    path = root / "per_mode_failure_trend.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_quality_dimensions(
    baseline_dims: dict[str, float],
    corrected_dims: dict[str, float],
    viz_dir: Path | None = None,
) -> Path:
    root = ensure_viz_dir(viz_dir)
    keys = list(QUALITY_KEYS)
    b = [baseline_dims.get(k, 0) for k in keys]
    c = [corrected_dims.get(k, 0) for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - w / 2, b, width=w, label="Baseline pass rate", color="#8172b3")
    plt.bar(x + w / 2, c, width=w, label="Corrected pass rate", color="#55a868")
    plt.xticks(x, keys, rotation=35, ha="right")
    plt.ylabel("Pass rate (1 = all pass)")
    plt.ylim(0, 1.05)
    plt.title("Quality dimension pass rates: baseline vs corrected")
    plt.legend()
    plt.tight_layout()
    path = root / "quality_dimensions_before_after.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_benchmark_vs_generated(
    benchmark_dims: dict[str, float],
    generated_dims: dict[str, float],
    viz_dir: Path | None = None,
) -> Path:
    root = ensure_viz_dir(viz_dir)
    keys = list(QUALITY_KEYS)
    bm = [benchmark_dims.get(k, 0) for k in keys]
    gen = [generated_dims.get(k, 0) for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - w / 2, bm, width=w, label="Benchmark sample (judge)", color="#dd8452")
    plt.bar(x + w / 2, gen, width=w, label="Generated (corrected)", color="#937860")
    plt.xticks(x, keys, rotation=35, ha="right")
    plt.ylabel("Pass rate")
    plt.ylim(0, 1.05)
    plt.title("Benchmark vs generated: quality dimension pass rates")
    plt.legend()
    plt.tight_layout()
    path = root / "benchmark_vs_generated.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_quality_dimensions_single_run(
    per_dimension_pass_rates: dict[str, float],
    run_label: str,
    viz_dir: Path | None = None,
) -> Path:
    """
    Pass rate (0–1) per quality dimension for one run (baseline or corrected).

    Use this when failure-mode charts are all zeros: quality can still vary
    (e.g. safety_specificity) because failure modes and quality scores are separate.
    """
    root = ensure_viz_dir(viz_dir)
    keys = list(QUALITY_KEYS)
    vals = [per_dimension_pass_rates.get(k, 0.0) for k in keys]
    x = np.arange(len(keys))
    plt.figure(figsize=(12, 6))
    plt.bar(x, vals, color=sns.color_palette("crest", n_colors=len(keys)))
    plt.xticks(x, keys, rotation=35, ha="right")
    plt.ylabel("Pass rate (fraction of items)")
    plt.ylim(0, 1.05)
    plt.title(
        f"Quality dimension pass rates — {run_label} "
        "(independent of failure-mode heatmaps; see README)"
    )
    plt.tight_layout()
    path = root / f"quality_dimensions_{run_label}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_most_problematic_count(
    records: Sequence[JudgeRecord],
    name: str,
    viz_dir: Path | None = None,
) -> Path:
    """Bar chart of items by number of simultaneous failure flags."""
    root = ensure_viz_dir(viz_dir)
    from analysis import count_failure_flags_per_item

    counts = count_failure_flags_per_item(records)
    buckets = Counter(counts)
    xs = sorted(buckets.keys())
    ys = [buckets[k] for k in xs]
    plt.figure(figsize=(8, 5))
    n_items = sum(ys) if ys else 0
    plt.bar([str(x) for x in xs], ys, color=sns.color_palette("rocket", n_colors=max(len(xs), 1)))
    plt.xlabel("Number of failure modes on same item (0 = none)")
    plt.ylabel("Count of items")
    plt.title("Distribution of simultaneous failure flags")
    if xs == [0] and n_items > 0:
        plt.text(
            0.5,
            0.95,
            f"All {int(n_items)} items had 0 failure-mode flags",
            ha="center",
            fontsize=10,
            transform=plt.gca().transAxes,
        )
    plt.tight_layout()
    path = root / f"failure_flag_distribution_{name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path
