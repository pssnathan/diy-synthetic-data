"""Phase 7: sample benchmark HF dataset for judge calibration."""

from __future__ import annotations

import logging
import random
from typing import Any

from datasets import load_dataset

from schemas import DIYRepairItem, RepairCategory

logger = logging.getLogger(__name__)

BENCHMARK_ID = "dipenbhuva/home-diy-repair-qa"


def load_benchmark_rows(
    n: int = 50,
    seed: int = 42,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Random sample of benchmark examples as dicts compatible with DIYRepairItem."""
    ds = load_dataset(BENCHMARK_ID, split=split)
    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)
    pick = indices[: min(n, len(indices))]
    rows: list[dict[str, Any]] = []
    for i in pick:
        ex = ds[i]
        row = {
            "question": ex["question"],
            "answer": ex["answer"],
            "equipment_problem": ex["equipment_problem"],
            "tools_required": list(ex["tools_required"]),
            "steps": list(ex["steps"]),
            "safety_info": ex["safety_info"],
            "tips": list(ex["tips"]),
            "category": ex["category"],
            "trace_id": ex.get("id") or f"bench_{i}",
        }
        rows.append(row)
    return rows


def validate_benchmark_compatibility(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that pass Pydantic (should be nearly all)."""
    good: list[dict[str, Any]] = []
    for row in rows:
        try:
            DIYRepairItem.model_validate(
                {k: v for k, v in row.items() if k in DIYRepairItem.model_fields}
            )
            good.append(row)
        except Exception as e:
            logger.warning("Benchmark row skipped: %s", e)
    return good
