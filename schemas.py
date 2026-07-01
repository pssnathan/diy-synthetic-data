"""Pydantic models for DIY Q&A items, judge output, and pipeline summaries."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RepairCategory = Literal[
    "appliance_repair",
    "plumbing_repair",
    "electrical_repair",
    "hvac_maintenance",
    "general_home_repair",
]


class DIYRepairItem(BaseModel):
    """Structured Q&A record matching the mini-project schema."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    equipment_problem: str = Field(..., min_length=1)
    tools_required: list[str] = Field(..., min_length=1)
    steps: list[str] = Field(..., min_length=3)
    safety_info: str = Field(..., min_length=1)
    tips: list[str] = Field(..., min_length=1)
    category: RepairCategory

    @field_validator("tools_required", "steps", "tips")
    @classmethod
    def non_empty_strings(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        return cleaned

    @model_validator(mode="after")
    def check_list_lengths(self) -> DIYRepairItem:
        if len(self.tools_required) < 1:
            raise ValueError("tools_required must have at least 1 item")
        if len(self.steps) < 3:
            raise ValueError("steps must have at least 3 items")
        if len(self.tips) < 1:
            raise ValueError("tips must have at least 1 item")
        return self


class QualityScores(BaseModel):
    """1 = pass, 0 = fail per dimension."""

    answer_coherence: int = Field(ge=0, le=1)
    step_actionability: int = Field(ge=0, le=1)
    tool_realism: int = Field(ge=0, le=1)
    safety_specificity: int = Field(ge=0, le=1)
    tip_usefulness: int = Field(ge=0, le=1)
    problem_answer_alignment: int = Field(ge=0, le=1)
    appropriate_scope: int = Field(ge=0, le=1)
    category_accuracy: int = Field(ge=0, le=1)


class JudgeOutputRaw(BaseModel):
    """Structured LLM judge response before post-processing."""

    trace_id: str = ""
    incomplete_answer: int = Field(ge=0, le=1)
    safety_violations: int = Field(ge=0, le=1)
    unrealistic_tools: int = Field(ge=0, le=1)
    overcomplicated_solution: int = Field(ge=0, le=1)
    missing_context: int = Field(ge=0, le=1)
    poor_quality_tips: int = Field(ge=0, le=1)
    quality_scores: QualityScores


class JudgeRecord(BaseModel):
    """Single evaluated item: failure modes + quality + derived flags."""

    trace_id: str
    incomplete_answer: int
    safety_violations: int
    unrealistic_tools: int
    overcomplicated_solution: int
    missing_context: int
    poor_quality_tips: int
    overall_failure: bool
    quality_scores: QualityScores
    quality_pass: bool


class GenerationLogEntry(BaseModel):
    """Per-item generation metadata (not part of published dataset)."""

    trace_id: str
    template_id: str
    category: RepairCategory
    structural_valid: bool
    structural_errors: list[str] = Field(default_factory=list)
    failure_modes_flagged: list[str] = Field(default_factory=list)
    timestamp_iso: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_used: str


class RunSummary(BaseModel):
    """Aggregated metrics for one pipeline run (baseline or corrected)."""

    run_name: str
    num_generated: int
    structural_pass_count: int
    structural_pass_rate: float
    overall_failure_rate: float
    per_mode_failure_rates: dict[str, float]
    quality_pass_rate: float
    per_dimension_quality_pass_rates: dict[str, float]
    items_with_three_plus_failures: int
    category_failure_rates: dict[str, float]
