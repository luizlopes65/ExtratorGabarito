# Milestones

## Objective

Optimize `extrair_table_profiling.py` incrementally while preserving extraction correctness and improving profiling signal quality.

## Success Criteria

- Extracted answers remain unchanged for the benchmark image used during validation.
- Total runtime is reduced measurably after Phase 1.
- Profiling output remains usable for hotspot analysis.
- Completion status is updated as work progresses.

## Benchmark and Validation Strategy

- Use the same input image for all comparisons.
- Record:
  - total execution time
  - top decorator hotspots
  - top cProfile cumulative hotspots
- Compare generated CSV outputs before and after each phase.
- If output changes unexpectedly, stop and investigate before continuing.

## Milestone 1 — Baseline and Guardrails

Status: Completed

- Review the current profiling script.
- Identify major hotspots and low-risk optimization targets.
- Define acceptance criteria and validation method.

## Milestone 2 — Phase 1 Quick Wins

Status: Completed

Scope:
- Remove duplicated `score_candidate(...)` calls in candidate scoring paths.
- Remove redundant reclustering in grid detection.
- Add explicit `OCR_MAX_WORKERS` instead of relying on default thread-pool sizing.
- Clean up clearly unused configuration only if confirmed safe.

Completed:
- Added `OCR_MAX_WORKERS = 4`.
- Removed duplicate `score_candidate(...)` calls in both candidate-selection paths.
- Removed redundant reclustering by returning raw positions from `extract_line_positions`.

Validation:
- Re-ran the profiling script on the benchmark image using `poetry run python extrair_table_profiling.py`.
- Observed total runtime of approximately `420 ms`.
- Confirmed successful CSV generation and coherent extraction output.
- Functional behavior appears unchanged for the benchmark run, but a formal before-vs-after baseline diff was not captured.

## Milestone 3 — Debug and Profiling Cost Control

Status: In Progress

Scope:
- Add `ENABLE_DEBUG_IMAGES`.
- Skip debug image writes and visualization generation when disabled.
- Reduce decorator coverage to coarse-grained functions by default.
- Keep a deeper profiling mode available for focused investigations.

Validation:
- Compare runtime with debug enabled vs disabled.
- Confirm debug artifacts still generate correctly when enabled.

## Milestone 4 — Cell Analysis Optimization

Status: Pending

Scope:
- Refactor `score_candidate` to use a local ROI instead of full-cell masks.
- Reduce temporary allocations in hot loops.
- Consider a non-copying crop path for read-only analysis.

Validation:
- Re-profile cell-analysis hotspots.
- Confirm extracted answers remain stable.

## Milestone 5 — OCR Path Optimization

Status: Pending

Scope:
- Benchmark sequential OCR versus bounded thread-pool sizes.
- Reuse OCR config construction where practical.
- Evaluate reducing repeated OCR preprocessing for cropped regions.

Validation:
- Compare runtime and OCR quality across configurations.
- Keep the simplest approach that improves speed without harming extraction.

## Milestone 6 — Optional Structural Improvements

Status: Pending

Scope:
- Evaluate row-level parallel cell analysis.
- Test connected-components as an alternative to contour scanning in band analysis.

Validation:
- Only keep changes that show measurable gains and no correctness regressions.

## Working Notes

Approved implementation scope right now:
- Implement only **Phase 1 quick wins** first.
- Update this file and task progress after each completed step.