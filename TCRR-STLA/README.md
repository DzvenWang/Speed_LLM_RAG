# TCRR-STLA v7.1.0 — C1 utility-calibrated candidate transfer

This is a **complete standalone v7.1.0 C1 research release** based on the supplied `TCRR-STLA v7.0.1` project. It requires sibling `Backbone` and `Base_SupportBench` projects and the user's original STLAformer checkpoints/data. **It does not include training data or Backbone checkpoints.** It does not use current-query future labels in inference.

## Scope and changes

- C1 now trains candidate utility against the **Normal-stage prediction** and correctly scores the **None → P0** full-activation fallback; the old P0-relative candidate targets with constant-zero None have been replaced.
- Ranking target temperature (`training.candidate_target_temperature`, default 0.25) is wired independently of predicted mixture temperature (`model.candidate_prediction_temperature`, default 1.0). The latter preserves original inference softness.
- The existing C1 leave-one-out marginal head uses four additional **candidate-set** context statistics: relative distance to weighted candidate mean, weighted set disagreement, direction agreement, and signed direction consistency. There is no quadratic K×K attention.
- Added masked NaN-safe train/evaluation losses, proper gradient accumulation for final partial groups, selection diagnostics comparing Learned with Uniform while holding None mass fixed, and run-identity SHA checks during stage resume.
- Default pipeline ends after **C1** and never automatically runs C2/Joint/Locked/Test. C2 code is preserved for a later version and can be explicitly requested with `--end-stage report` only after separate validation.
- New run namespace: `v7_1_0_c1`. **Never reuse an older v7_1 directory as the new run.**
- Includes optional **verified import** of v7.0.1 S0, teacher, graph and Replay caches. No old C1/C2 weight or optimizer is imported.

## Gadi installation and commands

Extract the ZIP under `/scratch/mh20/zw1165/Speed_LLM_RAG` so `TCRR-STLA/`, `Backbone/`, and `Base_SupportBench/` are siblings. Activate the existing `TFP` environment.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/TCRR-STLA:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
conda activate TFP

# Source unit tests, including None/utility/ranking/shape/masking tests.
python -m pytest -q TCRR-STLA/tests

# Full-sized CPU C1 forward/backward/optimizer contract, no GPU needed.
python -u TCRR-STLA/scripts/preflight_c1_v71.py \
    --nodes 388 --model-dim 64 --top-k 12 --hidden 64 --device cpu

# Replace these paths only if the official best.pt runs are different.
export STLA_SPEED="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export STLA_FLOW="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"

# Real Backbone/data preflight before queueing.
python -u TCRR-STLA/scripts/preflight_v7.py --target speed \
  --dataset TraffiDent_Net_06 --backbone-checkpoint "$STLA_SPEED"

# Submit dedicated C1-only run; wrapper reruns tests/preflights and identity checks.
bash TCRR-STLA/pbs/submit_v7.sh speed "$STLA_SPEED" 10
bash TCRR-STLA/pbs/submit_v7.sh flow "$STLA_FLOW" 10
```

The wrapper requests one V100 via `PBS`, and the allocated job additionally runs a small **CUDA+AMP C1 optimizer step** before beginning expensive Teacher/Replay work. Default training microbatch is 16, gradient accumulation 4, AMP enabled. A V100 32 GB memory-fit test with the full official data has **not** been run here; monitor peak memory and adjust `BATCH_SIZE` before submission if required.

## Reusing completed v7.0.1 caches (optional)

If the old v7.0.1 run has completely finished `s0`, `cache`, and `replay` stages and uses the **same** Target, Dataset, Seed, Backbone SHA, K, candidate retrieval settings, training rows, and graph algorithm, supply the old run directory explicitly. Example:

```bash
export REUSE_V7_RUN_DIR="$PWD/TCRR-STLA_exps/speed/incomplete/TraffiDent_Net_06/v7_1/formal/seed10"
bash TCRR-STLA/pbs/submit_v7.sh speed "$STLA_SPEED" 10
unset REUSE_V7_RUN_DIR
```

The submit wrapper creates and double-checks the new `run_identity.json`, then the importer verifies S0/teacher identities, graph identity, and **each replay file's SHA256** before linking (or copying) immutable source artifacts into the new run. Stage markers are written **last**; on validation failure **no qsub is performed**. The old run is never modified. The `normal` stage is freshly trained for v7.1.0 because the C1 marginal MLP input size changed (old full-model checkpoint would not load strictly). If identities differ, omit `REUSE_V7_RUN_DIR` and generate caches afresh; do **not** delete or rewrite manifests to bypass validation.

The source folder must be the actual v7.0.1 run directory. Do not point to a smoke run or an incomplete run.

## Smoke and manual execution

```bash
python -u TCRR-STLA/scripts/run_v7.py \
  --target speed --dataset TraffiDent_Net_06 \
  --backbone-checkpoint "$STLA_SPEED" --workers 0 --smoke

# Formal CPU run is forbidden unless runtime.allow_cpu_formal is explicitly enabled.
# End stage defaults to c1; run identity includes the selected end stage.
python -u TCRR-STLA/scripts/run_v7.py --target speed \
  --dataset TraffiDent_Net_06 --backbone-checkpoint "$STLA_SPEED" \
  --end-stage c1
```

`--smoke` creates an isolated `v7_1_0_c1/smoke` namespace with short limits/one epoch; smoke metrics are not formal evidence. The `train` and `val` teacher/replay caches use the existing Base_SupportBench protocol. No `test` split is read by the default C1 pipeline.

## Output and evaluation

The formal C1 results are written under:

```
TCRR-STLA_exps/<target>/<backbone_mode>/<dataset>/v7_1_0_c1/formal/seed10/
  run_identity.json
  s0.pt
  teacher_cache/{train,val}/
  paired_replay/{train,val}/
  best_normal.pt
  best_c1.pt
  last_c1.pt
  selection_c1.json
  STAGE_c1.done
```

`V71_C1_DIAGNOSTICS` logs Validation weight entropy, None mass, effective K, learned MAE, Uniform/Keep-None MAE, and absolute candidate utility prediction error. These are **report-only selection diagnostics**, not training inputs. Compare model skill on `Overall/Masked MAE`, `RMSE`, H12 and different fault families in follow-up studies. The release does **not** claim a measured accuracy gain over v7.0.1.

## Testing boundary

The project passed the included CPU unit tests and synthetic Speed/Flow stage pipelines in this environment. The fake synthetic vendor used by `tests/integration_synthetic_v7.py` is only an interface contract test; its metrics are not real forecasting evidence. No Gadi V100 execution or full real-data memory profile is possible in this environment. Always run the provided on-site preflight and check `V71_C1_PREFLIGHT_PASS` before treating a PBS allocation as ready.

See `docs/V7_1_C1_RELEASE.md` for migration assumptions and test boundaries.


## Add-on: Frozen one-time C1 CPU diagnostic and held-out TEST

The full project now includes `scripts/c1_frozen_analysis.py`, the CPU/GPU PBS evaluation templates and `pbs/submit_c1_frozen.sh`. These are **add-ons** that do not modify existing v7.1 training sources or Checkpoint identity. For every command, identity checks, Test validity rules, and output files, see **[docs/C1_FROZEN_DIAGNOSTIC_TEST.md](docs/C1_FROZEN_DIAGNOSTIC_TEST.md)**.
