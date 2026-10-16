# TCRR-STLA v1.0.0 — Traffic-Conditioned Residual Replay

Full **source project**, derived from the user's Mask_RAFT-STLA v2.3 portable project. Uses existing `Backbone/` and `Base_SupportBench/` siblings at runtime; neither data nor pretrained checkpoints are bundled. `Baseline/` is comparison-only.

## Implemented method

- **Train-only traffic candidate discovery**: a second retrieval route generates candidates from observed-history traffic signatures; temporal history+horizon exclusion is enforced. Existing multi-period Pearson route and forecast-conditioned shortlist remain. Caches are schema 24, isolated from v2.3.
- **Node-level conditional residual transfer**: observed-mask-only traffic features, candidate query-mask replay, learned traffic compatibility, and separately controlled historical normal/failure residuals.
- **Candidate and mixture utility**: train-label-supervised expected improvements, learned candidate bias, bounded risk-controlled historical residual correction, and explicit None routing.
- **Preservation**: previous STLAformer, frozen Teacher, counterfactual replay, progressive fine-tuning, training logs, best/last checkpoints, early stopping, inference, P0 candidate/Offline diagnostics, and paired retrieval audit.
- **Evaluation**: raw-value MAE (primary, chooses `best.pt`), RMSE, MAPE (percentage, |truth| > 1 following inherited Base_SupportBench-compatible protocol), severities s20/40/60/90, failure families, directly masked cells, H3/H6/H12.

The traffic state features are statistical indicators, **not** externally verified congestion stage labels; no physical conservation law is claimed. No Test future is used for retrieval or training.

## Dependency and environment

From the common repository containing `TCRR-STLA`, `Backbone`, `Base_SupportBench`:

```bash
conda activate TFP
python -u TCRR-STLA/scripts/preflight.py --target speed --backbone-checkpoint <BACKBONE_BEST_PT>
python -m pytest -q TCRR-STLA/tests/test_tcrr_traffic_v1.py
```

PBS runtime automatically finds the repository from `PBS_O_WORKDIR`/current directory and selects an activated TFP Python, `TFP_ENV_ROOT`, or `conda activate TFP`. No fixed scratch or gdata path.

## Independent formal training (Backbone-only initialization)

**No old Mask_RAFT / Mask_RAST Full checkpoint is read or used.** The executable train entrypoint has no `--initialize-from-v23` option. Do **not** set `V23_CHECKPOINT` or `RESUME_CHECKPOINT` for new formal runs. The PBS script fails fast if either inherited variable is present.

From the repository root, activate TFP and use the validated four-job submission script in the repository root (`submit_TCRR_PCFT_four.sh`), or submit this project with:

```bash
qsub -V -N TCRR_speed -v TARGET=speed,SEED=10,BACKBONE_MODE=incomplete,ABLATION=full,BACKBONE_CHECKPOINT=Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt TCRR-STLA/pbs/formal.pbs
qsub -V -N TCRR_flow -v TARGET=flow,SEED=10,BACKBONE_MODE=incomplete,ABLATION=full,BACKBONE_CHECKPOINT=Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt TCRR-STLA/pbs/formal.pbs
```

Use the exact target-matched pre-trained `Backbone_exps` checkpoints, and verify that the checkpoints exist. Output directories are independent for TCRR-STLA; `Base_SupportBench` and `Backbone` inputs are shared.

New project only: no `Mask_RAFT-STLA` or `Mask_RAST-STLA` path dependency, and no old Full checkpoint migration helper.

## Evaluation and experiment controls

Main metric **MAE**, auxiliary **RMSE/MAPE**. Full run automatically evaluates s20/40/60/90. `ABLATION=full` runs the complete new model, not a reduced control. PBS requests gpuvolta, V100, 12 CPU, 32 GB, 48 h per job. Preflight and real Gadi smoke/formal runs remain necessary.
