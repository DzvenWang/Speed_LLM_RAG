# Backbone v1.0.1 — PatchSTG on Protocol v2.0.1

This Backbone is the numerical reference for the current TraffiDent incomplete-sensing study. It intentionally removes the old multi-model screening workflow from the main execution path and uses **PatchSTG only**.

## Frozen upstream contracts

- TraffiDent-SupportBench benchmark: **v1.2.6**
- Protocol: **v2.0.1**
- ExperimentCommon shared contract: **v1.1.0**
- ExperimentCommon implementation: **v1.1.1**
- Cache schema: **2**

Main training uses `protocol_core5_s40`:

- development train: 12,000 = 2,400 × 5 families
- development val: 5,000 = 1,000 × 5 families
- formal train: 50,000 = 10,000 × 5 families
- formal val: 5,000 = 1,000 × 5 families

Frozen paired evaluation caches are supported at 20/40/60/90%, each with 5,000 queries = 1,000 × 5 families.

## Portable project layout

Place `Backbone/` beside `ExperimentCommon/`, `Protocol/`, `TraffiDent_SupportBench/`, `ExperimentCache/`, etc. The root is inferred automatically. To override it:

```bash
export SPEED_LLM_RAG_ROOT=/path/to/Speed_LLM_RAG
```

PowerShell:

```powershell
$env:SPEED_LLM_RAG_ROOT = "C:\path\to\Speed_LLM_RAG"
```

## Gadi PatchSTG PBS

For the standard formal seed-10 run on Gadi:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
mkdir -p /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone_logs
qsub pbs/run_patchstg.pbs
```

The PBS script defaults to `STAGE=formal` and `SEED=10`, uses the TFP Python executable directly, and writes one timestamped live log to `Backbone_logs/PatchSTG.<stage>.seed<seed>.<jobid>.<timestamp>.live.log`. Override the seed with `qsub -v STAGE=formal,SEED=20 pbs/run_patchstg.pbs`.

## 1. Check caches

```powershell
python scripts\preflight.py --stage development
python scripts\preflight.py --stage formal
python scripts\preflight.py --stage evaluation --severity 20
python scripts\status.py
```

Evaluation is available as soon as the frozen evaluation cache passes hash/contract validation. No Protocol access-grant file is required.

## 2. Development

```powershell
python scripts\run_experiment.py --stage development --seed 10
```

Logs are written to `Backbone_logs/PatchSTG.development.seed10.<timestamp>.log`.

## 3. Formal training

Run seeds sequentially on one GPU:

```powershell
python scripts\run_experiment.py --stage formal --seed 10
python scripts\run_experiment.py --stage formal --seed 20
python scripts\run_experiment.py --stage formal --seed 30
python scripts\summarize_formal.py
```

This creates:

- `Backbone_exps/backbone_formal_runs.csv`
- `Backbone_exps/backbone_formal_summary.csv`
- `Backbone_exps/FROZEN_BACKBONE.json`

Seed 10 is the downstream reference checkpoint. All three formal checkpoints are retained for final mean/std evaluation.

## 4. Final evaluation

Evaluation caches may be consumed directly once frozen and valid. `run_evaluation.py` does not require a Protocol access-grant file.

Reference checkpoint only:

```powershell
python scripts\run_all_evaluations.py --reference-only
```

All three formal seeds × four severities:

```powershell
python scripts\run_all_evaluations.py
```

Or individually:

```powershell
python scripts\run_evaluation.py --severity 20 --seed 10
python scripts\run_evaluation.py --severity 40 --seed 10
python scripts\run_evaluation.py --severity 60 --seed 10
python scripts\run_evaluation.py --severity 90 --seed 10
```

The summary is written to `Backbone_exps/backbone_evaluation_summary.csv`.

## Important contract changes from Backbone v0.6.0

- public stage `screening` is replaced by `development`;
- shared contract requirement is 1.2.0, cache schema is 2;
- cache lookup is Protocol/severity/stage/hash aware;
- formal train size is exactly 50,000, balanced 10,000 per family;
- evaluation supports paired s20/s40/s60/s90 caches;
- standard evaluation consumers require frozen/hash-valid evaluation caches but do not enforce a Protocol access gate;
- old `ExperimentCache/TraffiDent_Net_06/development` and `/formal` paths are not used.

The official PatchSTG source is pinned under `vendor/PatchSTG` and verified by `scripts/source_lock.py`.

## Explicit evaluation of an existing PatchSTG checkpoint

To evaluate a historical or ad-hoc `best.pt` without changing the official
`FROZEN_BACKBONE.json`, use `scripts/evaluate_checkpoint.py` or the
cross-platform `scripts/run_checkpoint_evaluations.py`. GADI submission is provided by
`pbs/evaluate_checkpoint_severity.pbs` and `submit_checkpoint_evaluations.sh`.

Local `run_checkpoint_evaluations.py` automatically writes a timestamped live log to `Speed_LLM_RAG/Backbone_logs/`. Every emitted line is timestamped. GADI PBS evaluation keeps one timestamped `.live.log` per severity in the same directory.
See `OLD_CHECKPOINT_EVALUATION.md` for the exact local and GADI commands.

## Historical PatchSTG spatial layout

For checkpoints created before v1.0.5, do not bypass a `spatial layout mismatch`. Run `scripts/recover_patchstg_layout.py` to reconstruct the exact old KDTree/reorder layout from an archived SupportBench copy. See `LEGACY_LAYOUT_RECOVERY.md`. New v1.0.5 checkpoints embed and sidecar the complete layout automatically.
