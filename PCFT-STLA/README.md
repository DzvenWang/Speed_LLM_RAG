# PCFT-STLA v1.0.0 — Propagation-Compatible Future Transfer

Full **source project**, derived from the user's Mask_RAST-STLA v2.3 portable project. Needs sibling `Backbone/` and `Base_SupportBench/` and externally stored checkpoints/data. `Baseline/` is comparison-only.

## Implemented method

- **Train-only directed lag association**: retain v2.3 undirected spatial graph for its region/state encoder, build a second signed directed predictive-lag graph with lag weights and confidence in a versioned Train-only cache.
- **Independent propagation retrieval**: append a propagation-signature key to temporal/spatial/region memory. Memory keys derive only from the historic Train *inputs*, never Query/Test futures. Temporal history+horizon overlap exclusion remains mandatory.
- **Horizon-aware support**: for each future step calculate which observed source-node history can arrive within empirical lag. No future observation is ever read.
- **Lag-aligned candidate innovations**: signed sparse neighbor-to-target temporal transfer; combine with the existing anchor/neighbor/seasonal-innovation evidence using a zero-initialized learnable bounded alignment coefficient. The old alignment is unchanged at initialization when using the same candidate pool.
- **Transfer utility and control**: learned propagation-dependent candidate score, supervised expected candidate gain, original None and Risk Gate, v2.3 query-memory contrastive alignment and progressive training retained.
- **Evaluation**: raw-value MAE (primary, selects `best.pt`), RMSE, MAPE (percentage, |truth| > 1), severities s20/40/60/90, failure family and directly masked metrics, inherited diagnostics and P0 Probe.

The directed edge coefficients are **lagged statistical predictive associations** estimated only from Train, not a measured road graph, true shockwave velocity or causal effect. No future label of the current Query is used at inference.

## Dependency and environment

From repository root:

```bash
conda activate TFP
python -u PCFT-STLA/scripts/preflight.py --target flow --backbone-checkpoint <BACKBONE_BEST_PT>
python -m pytest -q PCFT-STLA/tests/test_pcft_propagation_v1.py
```

Portable `pbs/runtime.sh` locates repository root and TFP Python automatically. No fixed user-specific absolute paths.

## Independent formal training (Backbone-only initialization)

**No old Mask_RAFT / Mask_RAST Full checkpoint is read or used.** The executable train entrypoint has no `--initialize-from-v23` option. Do **not** set `V23_CHECKPOINT` or `RESUME_CHECKPOINT` for new formal runs. The PBS script fails fast if either inherited variable is present.

From the repository root, activate TFP and use the validated four-job submission script in the repository root (`submit_TCRR_PCFT_four.sh`), or submit this project with:

```bash
qsub -V -N PCFT_speed -v TARGET=speed,SEED=10,BACKBONE_MODE=incomplete,ABLATION=full,BACKBONE_CHECKPOINT=Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt PCFT-STLA/pbs/formal.pbs
qsub -V -N PCFT_flow -v TARGET=flow,SEED=10,BACKBONE_MODE=incomplete,ABLATION=full,BACKBONE_CHECKPOINT=Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt PCFT-STLA/pbs/formal.pbs
```

Use the exact target-matched pre-trained `Backbone_exps` checkpoints, and verify that the checkpoints exist. Output directories are independent for PCFT-STLA; `Base_SupportBench` and `Backbone` inputs are shared.

New project only: no `Mask_RAFT-STLA` or `Mask_RAST-STLA` path dependency, and no old Full checkpoint migration helper.

## Evaluation and experiment controls

Main metric **MAE**, auxiliary **RMSE/MAPE**. Full run automatically evaluates s20/40/60/90. `ABLATION=full` runs the complete new model, not a reduced control. PBS requests gpuvolta, V100, 12 CPU, 32 GB, 48 h per job. Preflight and real Gadi smoke/formal runs remain necessary.

## v1.0.1 resource safety hotfix (2026-10-18)

For the existing Speed/Flow `best.pt` produced before this hotfix, **do not
retrain** just to resolve an evaluator OOM or DataLoader SIGBUS. See
[`docs/GADI_EVALUATION_RECOVERY_v1.0.1.md`](docs/GADI_EVALUATION_RECOVERY_v1.0.1.md)
for checkpoint checks and separate Gadi `qsub` commands. The original trained
checkpoint schema remains `version=1.0.0`; the source package is `1.0.1`.
