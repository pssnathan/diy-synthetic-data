# Prompt iteration log (template)

Document each cycle after you run the pipeline and inspect `output/*_summary.json` and `visualizations/`.

### Iteration 1: Baseline
- **Date**: YYYY-MM-DD
- **Change**: Baseline prompts only (`prompts.py` → `BASELINE_TEMPLATES`).
- **Hypothesis**: Lighter instructions surface enough failures to measure and learn from.
- **Result**: Overall failure rate = ___%; quality pass rate = ___%.
- **Decision**: Establish baseline / adjust templates.
- **Next step**: Inspect failure co-occurrence heatmap; pick dominant failure mode.

### Iteration 2: Targeted correction (example)
- **Date**: YYYY-MM-DD
- **Change**: Strengthened safety and narrative requirements in `electrical_repair` corrected template.
- **Hypothesis**: Reduces `safety_violations` and improves `safety_specificity`.
- **Result**: Per-mode rates: ___; quality dimensions: ___.
- **Decision**: Keep / revert / modify.
- **Next step**: Address next-highest failure mode.

### Iteration 3: Context and steps
- **Date**: YYYY-MM-DD
- **Change**: (Describe edits to `CORRECTED_TEMPLATES` or judge calibration.)
- **Hypothesis**: …
- **Result**: …
- **Decision**: …
- **Next step**: …

### Iteration 4: Judge calibration
- **Date**: YYYY-MM-DD
- **Change**: Adjusted `JUDGE_SYSTEM` in `judge.py` after benchmark sample showed miscalibration on dimension ___.
- **Hypothesis**: Judge should pass ≥95% of benchmark items per dimension after calibration.
- **Result**: Benchmark worst-dimension pass rate = ___%.
- **Decision**: …
- **Next step**: …

---

| Iteration | Change (short) | Overall failure rate | Quality pass rate | Decision |
|-----------|------------------|----------------------|-------------------|----------|
| 1 | Baseline | | | Establish baseline |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
