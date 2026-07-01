"""Unit tests for Pydantic schema validation (no API calls)."""

import pytest

from schemas import DIYRepairItem, JudgeRecord, QualityScores


def test_valid_item():
    item = DIYRepairItem(
        question="How do I fix a dripping faucet?",
        answer="Turn off the water under the sink. Disassemble the handle and replace the washer. "
        * 20,
        equipment_problem="Kitchen faucet spout drip",
        tools_required=["wrench"],
        steps=["Turn off water", "Remove handle", "Replace washer and reassemble"],
        safety_info="Turn off the supply before work. " * 5,
        tips=["Use plumber tape on threads"],
        category="plumbing_repair",
    )
    assert item.category == "plumbing_repair"


def test_invalid_too_few_steps():
    with pytest.raises(Exception):
        DIYRepairItem(
            question="q",
            answer="a",
            equipment_problem="e",
            tools_required=["t"],
            steps=["only one", "two"],
            safety_info="Turn off the supply before work. " * 5,
            tips=["tip"],
            category="plumbing_repair",
        )


def test_judge_record_quality_pass():
    qs = QualityScores(
        answer_coherence=1,
        step_actionability=1,
        tool_realism=1,
        safety_specificity=1,
        tip_usefulness=1,
        problem_answer_alignment=1,
        appropriate_scope=1,
        category_accuracy=1,
    )
    r = JudgeRecord(
        trace_id="t1",
        incomplete_answer=0,
        safety_violations=0,
        unrealistic_tools=0,
        overcomplicated_solution=0,
        missing_context=0,
        poor_quality_tips=0,
        overall_failure=False,
        quality_scores=qs,
        quality_pass=True,
    )
    assert r.quality_pass
