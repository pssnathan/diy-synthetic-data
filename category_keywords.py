"""Rule-based hints for category accuracy (Q8); used with LLM judge."""

from __future__ import annotations

from schemas import RepairCategory

_KEYWORDS: dict[RepairCategory, tuple[str, ...]] = {
    "appliance_repair": (
        "washer",
        "washing machine",
        "dryer",
        "dishwasher",
        "refrigerator",
        "fridge",
        "oven",
        "range",
        "microwave",
        "appliance",
    ),
    "plumbing_repair": (
        "plumb",
        "pipe",
        "faucet",
        "toilet",
        "drain",
        "clog",
        "leak",
        "water heater",
        "p-trap",
        "sink",
    ),
    "electrical_repair": (
        "outlet",
        "receptacle",
        "breaker",
        "wire",
        "electrical",
        "switch",
        "fixture",
        "lighting",
        "circuit",
        "ground",
    ),
    "hvac_maintenance": (
        "hvac",
        "furnace",
        "air conditioner",
        "ac unit",
        "thermostat",
        "filter",
        "duct",
        "vent",
        "heating",
        "cooling",
    ),
    "general_home_repair": (
        "drywall",
        "door",
        "window",
        "floor",
        "baseboard",
        "trim",
        "paint",
        "caulk",
        "wall",
        "hinge",
    ),
}


def _text_blob(question: str, answer: str, equipment_problem: str) -> str:
    return f"{question} {answer} {equipment_problem}".lower()


def keyword_matches_category(
    category: RepairCategory, question: str, answer: str, equipment_problem: str
) -> bool:
    """Heuristic: at least one domain keyword appears in the combined text."""
    text = _text_blob(question, answer, equipment_problem)
    for kw in _KEYWORDS[category]:
        if kw in text:
            return True
    # Very short/generic text may not match; allow pass to avoid false negatives
    if len(text.strip()) < 40:
        return True
    return False


def score_category_keywords(category: RepairCategory, text: str) -> float:
    """Simple 0-1 score: fraction of keyword hits (for diagnostics)."""
    t = text.lower()
    hits = sum(1 for kw in _KEYWORDS[category] if kw in t)
    return min(1.0, hits / 3.0) if _KEYWORDS[category] else 0.0
