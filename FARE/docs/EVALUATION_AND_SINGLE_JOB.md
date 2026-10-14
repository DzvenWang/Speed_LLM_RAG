# FARE v3.0.2: post-training evaluation scope and one-GPU PBS job

## Default

- Validation **during training** remains enabled for epoch monitoring and best.pt checkpoint selection.
- Validation **after training** is OFF by default.
- Post-training evaluation and fixed-checkpoint internal attribution run only Test s20, s40, s60, s90.
- Counterfactual audit runs on the selected `analysis.counterfactual_split` (default `s40`); Test labels are used **only for post-hoc research diagnostics**. Do not update weights, tune hyperparameters, or select checkpoints based on the Test audit.
- Set `evaluation.include_val: true` in YAML or pass `--eval-val` to include Val in post-training evaluation/analysis. `--test-only` explicitly overrides YAML to Test only.

## PBS

`python -u FARE/scripts/submit_gadi.py --config FARE/configs/pilot.yaml --target speed --seed 10 --checkpoint /abs/best.pt`
submits exactly one job running `FARE/pbs/full_pipeline.pbs`. That job uses 1 V100 and 12 CPUs to sequentially perform full training, Test evaluation, and internal analysis.

No separate `FARE/pbs/train.pbs` or `FARE/pbs/analysis.pbs` are needed.

Important: previously submitted PBS jobs keep their original submission/dependency structure. Cancelling a running training job loses incomplete work unless resumable checkpoints are used. Inspect `qstat` before replacing code in a running environment.

## Output

- `evaluation/performance_summary.csv` has only `s20`, `s40`, `s60`, `s90`, and computed averages in default mode, no post-training `val` line.
- `analysis/module_contributions.csv`, traffic/risk/interaction diagnostics contain only Test splits by default.
- `analysis/counterfactual/candidate_counterfactual.csv` records which split the audit came from.
- There is no additional training or external model comparison.

## Time budget

Four full test severities with sixteen internal interventions, plus counterfactual audit, can exceed a 48-hour one-job walltime on a V100. Best and last checkpoints are written during training; analysis/evaluation have separate output files. The single-GPU request is enforced, not a promise of completion within the walltime.
