# Phase 1 execution & sign-off record

Date: 2026-11-24

## Commands executed

~~~bash
rm -rf artifacts/mock
python scripts/run_mock.py --config configs/mock.yaml
python scripts/run_mock.py --config configs/mock.yaml
PYTHONPATH=src python -m pytest -v
~~~

## Acceptance checklist & verification results

1. **Dataset scale, diversity & visual QA (`24/24` images inspected)**:
   - 24 images (`768x768` true 3-channel `RGB` PNGs) and 48 questions (`24` chart + `24` geometry; `1` direct + `1` multi-step per image) across **12 genuine sub-templates** (`6` chart subtypes: 4-bar 5-unit grid, 4-barh 5-unit grid, 5-point line 5-unit grid, 2-series grouped bar, 2-segment stacked bar, 5-bar target threshold; `6` geometry subtypes: scalene triangle, right-angle square symbol, isosceles tick-mark triangle, exterior angle, parallel line at apex, two-triangle Cevian).
   - Full visual inspection across `chart-000..011.png` and `geometry-000..011.png` confirmed zero label-line collisions, exact grid alignment on grid-read charts, clean angle-bisector arcs/markers, and zero standalone numeric answer leakage in prompt text.
   - `validate_dataset` independently recomputes all 48 answers from generator metadata, verifies pixel-matrix SHA256 hashes, enforces Euclidean vector-angle consistency (`< 1e-5` deg tolerance) on all geometry vertices, and checks portable relative `image_path` (`images/<id>.png`).
   - Variable scale (`n_per_type=60` for `synth-240`) supported without hardcoded count limits.

2. **Resumability, atomic writes & fault injection**:
   - First mock invocation wrote `48` rows; second invocation wrote `0` new rows (`config_id=181ef2aca055`).
   - `storage.py` enforces `os.fsync` append writes, atomic temp-file + `os.replace` writes for JSON/CSV/Markdown manifests and reports, automatic recovery of both unterminated and NUL/corrupt trailing crash lines, and hard failure (`ValueError`) on mid-file corruption.

3. **Mutually exclusive 5-way outcome accounting & leakage controls**:
   - `grading.py` and `metrics.py` separate `correct`, `wrong_answers`, `parse_failures`, `truncated` (`finish_reason=length`), and `backend_errors` (`sum == n`).
   - `runner.py` dispatches via `create_backend(config)`, blocks unpinned `model_revision` placeholders on non-mock backends, and supports `image_mode` in `("correct", "blank", "mismatched")` for Phase 3 visual leakage gates.

4. **Test suite & reporting**:
   - Pytest: **13/13 passed** (`test_config.py`, `test_data.py`, `test_grading.py`, `test_pipeline.py`, `test_resume.py`).
   - Report artifacts (`summary.json`, `summary.csv`, `report.md`, `pareto_mock.png`) generated and stamped `MOCK / TEST DATA — NOT REAL MODEL PERFORMANCE`.
   - Zero GPU compute, large model downloads, or external proxy calls used.

## Sign-off status

**Phase 1 is formally signed off and frozen (`schema_version: 2`)**, ready for merge into local `main` and progression to Phase 2 (pinning `TML-d24-varloop-research` SHA and running the single-image Colab GPU `loop=1 vs 3` Transformers probe).
