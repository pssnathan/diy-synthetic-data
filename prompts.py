"""
Prompt templates for DIY repair Q&A generation.

BASELINE_*: intentionally lighter instructions to establish a measurable failure
rate for learning/demonstration (per project success metrics).

CORRECTED_*: data-driven improvements aligned with the 8 quality dimensions
and 6 failure modes (replace or extend based on your own failure analysis).
"""

from __future__ import annotations

from typing import Literal

from schemas import RepairCategory

PromptVariant = Literal["baseline", "corrected"]

# --- Baseline: shorter instructions (more room for judge failures) ---
BASELINE_TEMPLATES: dict[RepairCategory, str] = {
    "appliance_repair": """You are a home appliance help assistant.
Generate ONE realistic DIY repair Q&A for a common homeowner appliance issue (washer, dryer, fridge, oven, dishwasher).
Output must follow the schema: question, answer, equipment_problem, tools_required, steps, safety_info, tips, category.
Keep the answer fairly short. List a few tools and 3+ steps. Include some safety text and one tip.
Category must be exactly: appliance_repair""",
    "plumbing_repair": """You are a plumbing DIY helper.
Create ONE Q&A about a typical plumbing problem (leak, clog, fixture).
Include question, answer, equipment_problem, tools_required, steps, safety_info, tips, category.
Answer can be brief. Steps should be numbered ideas, not necessarily very detailed.
Category must be exactly: plumbing_repair""",
    "electrical_repair": """You are helping with small home electrical jobs (outlets, switches, fixtures) at homeowner level.
Produce ONE Q&A. Mention turning off power somewhere in the answer.
Use fields: question, answer, equipment_problem, tools_required, steps, safety_info, tips, category.
Category must be exactly: electrical_repair""",
    "hvac_maintenance": """Write ONE HVAC maintenance or basic troubleshooting Q&A (filters, thermostat, vents).
Include all required fields. Keep instructions general.
Category must be exactly: hvac_maintenance""",
    "general_home_repair": """Write ONE general home repair Q&A (drywall, door, window, floor, basic carpentry).
Include all required fields. Answer may be generic.
Category must be exactly: general_home_repair""",
}

# --- Corrected: explicit bar for narrative answer, specificity, tools, safety length, tips ---
CORRECTED_TEMPLATES: dict[RepairCategory, str] = {
    "appliance_repair": """You generate training data for a Home DIY repair assistant.

Category: appliance_repair (refrigerators, washers, dryers, dishwashers, ovens).

Requirements:
- question: realistic homeowner question.
- equipment_problem: specific symptom (not generic "broken appliance").
- answer: ONE cohesive narrative (roughly 600–1200 characters) weaving tools, safety, steps, and tips—NOT a bulleted paste of the other fields.
- tools_required: only common homeowner tools or under-$50 hardware-store items (no trade-only gear).
- steps: at least 3 concrete steps with observable outcomes, quantities, or measurements where relevant; avoid vague words like "properly" or "as needed" without detail.
- safety_info: at least 80 characters, naming the specific hazard AND the specific precaution (never only "be careful").
- tips: 1+ tips with non-obvious, task-specific advice not duplicated in the steps.
- category: must be exactly "appliance_repair".

The repair must stay within typical DIY scope; if professional service is truly required, say so clearly instead of unsafe instructions.""",
    "plumbing_repair": """You generate training data for a Home DIY repair assistant.

Category: plumbing_repair (leaks, clogs, fixtures, pipes).

Requirements:
- question, equipment_problem, answer as a unified narrative (600–1200 chars) that directly fixes the stated problem.
- tools_required: homeowner-realistic tools only.
- steps: specific, executable steps (3+), with specifics where it matters.
- safety_info: >=80 chars, hazard-specific (water damage, cuts, contamination, etc.).
- tips: non-obvious, not a restatement of steps.
- category: exactly "plumbing_repair".""",
    "electrical_repair": """You generate training data for a Home DIY assistant for SAFE homeowner electrical work only
(outlets, switches, simple fixtures—not panel replacement, not gas).

Category: electrical_repair.

Requirements:
- answer: narrative 600–1200 chars integrating safety with procedure.
- Explicitly name hazards (shock, arc flash) and precautions (breaker off, test with non-contact tester, etc.).
- safety_info: >=80 chars, specific to this repair.
- tools: typical homeowner electrical tools (non-contact voltage tester, insulated screwdriver, etc.), no exotic trade tools.
- steps: 3+ specific steps; if scope is beyond DIY, refuse step-by-step hazardous work and recommend a licensed electrician.
- tips: practical, task-specific.
- category: exactly "electrical_repair".""",
    "hvac_maintenance": """You generate training data for a Home DIY assistant.

Category: hvac_maintenance (filters, thermostat, vents, basic troubleshooting).

Requirements:
- Narrative answer 600–1200 chars, coherent and on-topic.
- tools_required: realistic for homeowners.
- steps: 3+ actionable steps.
- safety_info: >=80 chars (electrical, ladder, dust, rotating parts as relevant).
- tips: useful and specific.
- category: exactly "hvac_maintenance".""",
    "general_home_repair": """You generate training data for a Home DIY assistant.

Category: general_home_repair (drywall, doors, windows, flooring, basic carpentry).

Requirements:
- Answer must address the exact equipment_problem in a 600–1200 character narrative.
- tools_required: homeowner-appropriate.
- steps: 3+ concrete steps.
- safety_info: >=80 chars, specific hazards for this task.
- tips: non-obvious, helpful.
- category: exactly "general_home_repair".""",
}


def get_template(category: RepairCategory, variant: PromptVariant) -> str:
    templates = BASELINE_TEMPLATES if variant == "baseline" else CORRECTED_TEMPLATES
    return templates[category]


def all_categories() -> list[RepairCategory]:
    return list(BASELINE_TEMPLATES.keys())
