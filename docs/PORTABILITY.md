# Portability notes

This repository preserves two kinds of executable material:

1. **Portable Python entry points** — accept data/model/output paths as command-line arguments and are the recommended interface.
2. **Frozen launcher snapshots** — shell scripts and timing records that preserve the exact server layout used for an audited run.

Some frozen launchers contain absolute `/root/...`, `/home/...`, or historical local paths. They are retained as provenance and should not be executed unchanged on another machine. They do not contain credentials.

## Recommended portable entry points

- Chinese core and formal ablations: `src/chinese_core/*.py`
- Reviewer controls: `experiments/reviewer_controls/run_reviewer_controls.py`
- Follow-up diagnostics: `experiments/reviewer_controls/run_followup_diagnostics.py`
- Recent-baseline evaluator: `experiments/recent_baselines/evaluate_cail_baselines.py`
- DELTA adaptation: `experiments/recent_baselines/run_delta_cail.py`
- LCG adaptation: `experiments/recent_baselines/run_lcg_blackbox.py`
- Zero-Shot LCS: `experiments/recent_baselines/run_zero_shot_lc.py`
- IN-Abs phases: `src/english_inabs/run_inabs_phase*.py`

## Frozen path-dependent material

- `experiments/cost/` launchers and stored timing manifests;
- `experiments/runtime/` shell launchers;
- `experiments/inabs_extension/run_*.sh` and resume scripts;
- `experiments/recent_baselines/start_*.sh`, full-pipeline launchers, and server-specific subset scripts;
- old document-generation utilities under `tools/`.

To port a frozen launcher, copy it to a new filename, replace fixed roots with explicit arguments or task-specific environment variables, and record the change as a new experiment configuration. Do not silently rewrite the frozen provenance version.
