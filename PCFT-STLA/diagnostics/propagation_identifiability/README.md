# v1.3 hotfix: Compute-node storage mount (2026-11-02)

**Verified against successful historical PBS source scripts** in the supplied
`Backbone(20261026-055736).zip` and `PCFT-STLA_v6.0.0_AuditedDualGraph_Full_Source.zip`.
Both successful jobs declare `#PBS -l storage=scratch/mh20+gdata/mh20`, but v1.2
diagnostic accidentally omitted it in **both** PBS files. The compute-node
`FATAL_TFP_PYTHON_NOT_EXECUTABLE` was therefore consistent with an unmounted
`/g/data/mh20` environment, despite login-node preflight passing.

v1.3 adds the exact storage resources to GPU and CPU PBS and a
compute-node project-files / checkpoint check. It also makes the allocated
GPU preflight run an actual STLA TRAIN+VAL inference **on CUDA** using
`--preflight-gpu`, rather than silently forcing CPU. The standard
`pre_submit_check.sh` continues running CPU-only on the login node.

**Required**: run `bash .../scripts/pre_submit_check.sh` again after unpacking,
then verify the PBS directives (unlike previous versions, pre-submit tests
assert the storage request is present). Only then submit the jobs.

**Important limitation:** running PBS as `bash script.pbs` does not cause PBS to
mount storage. A dry-run can verify directives and error propagation but only a
real `qsub` compute-node allocation can verify the mount and CUDA runtime.

---

# v1.2 correction (2026-11-02)

The v1.1 `pre_submit_check.sh` rejected otherwise valid TRAIN data because its new
preflight erroneously asserted `f.shape == (1,6)` while
`edge_horizon_features()` generates **seven** columns (intercept + three pairs).
This is a diagnostic-code assertion bug, **not evidence of a corrupt TRAIN
cache**. The same wrong assertion affected VAL and would also fail GPU PBS
startup. v1.2 removes duplicated feature-width assumptions, audits real TRAIN
and VAL through the shared `audit_forecast_arrays()` routine, and includes
regression tests that exercise exactly one sample with 388 nodes plus missing
source values, invalid labels, several lags/horizons, and malformed shape.

**Before resubmitting GPU:** unpack over v1.1, run
`bash PCFT-STLA/diagnostics/propagation_identifiability/scripts/pre_submit_check.sh`
and require the final `READY_FOR_GPU_PBS` marker. `PREFLIGHT_TRAIN_FEATURES_PASS`
and `PREFLIGHT_VAL_FEATURES_PASS` should both report `local_feature_dim: 7`.

---

# PCFT — Net-06 Propagation Identifiability Diagnostic (v1.2, feature-contract and PBS audited)

**研究问题：** TraffiDent_Net_06 的 TRAIN 数据中，有多少条物理有向候选关系同时满足：方向优势、时滞稳定，以及相对于冻结 STLAformer 预测的额外预测收益？

This is an independent diagnostic under `PCFT-STLA/diagnostics/propagation_identifiability/` and is **not** a new PCFT model. It does not read old PCFT run artifacts and never modifies Backbone, Base_SupportBench, their model weights, or the immutable dataset. It writes only JSON/CSV and logs. It **never opens TEST**.

## What the counts mean

1. `physical_directed_candidates`: audited physical candidate arcs, both road-forward and disturbance-reverse. Expected graph may be 3,486, but the script reads the *current* raw input.
2. `discovery_stable_lag_edges`: early 65% of TRAIN only, raw-VALID speed/flow change lead/lag correlation, six time blocks, observed-pair filtering, sign and lag +/-1 stability.
3. `discovery_direction_and_lag_edges`: additionally requires absolute correlation in this direction to exceed the reverse direction by >=0.03. **Predictive temporal asymmetry, not causal wave direction.**
4. `train_gain_qualified`: frozen **official STLA**, predictions on incomplete s40 TRAIN sensing views, plus **small ridge residual correction** based on a source change, compared on identical observed-source rows against an equal-complexity local-only residual correction. Requires improvement >0.2% over the local-only corrected baseline and positive improvement over the uncorrected STLA, with >=3/4 heldout time groups positive. The regression is fit **only** on early TRAIN windows; evaluated on untouched later TRAIN windows.
5. `confirmed_count`: the subset of train-qualified relations that independently meets these fixed rules on the **official VAL** s40 views; never used to select candidate/lag or fit coefficients. This is a more credible heldout result because the official STLA was trained on the full original TRAIN cache. No Test is used.

**Critical timing rule:** for discovered lag L (5-minute steps), forecast horizon h may use historical observed source change at time `origin+h-L` ONLY if `1 <= h <= L`. Therefore this diagnostic evaluates H1..HL, not full H1..H12. For h>L, source future changes are unavailable unless a propagation model generates them recursively; do NOT use future ground truth as features. This makes the diagnostic a conservative short-horizon measure, **not an estimate of full PCFT headroom**.

**Interpretation limits:** The correlation discovery uses original raw-VALID TRAIN readings, an optimistic observation upper bound not the exact s40 view. The residual correction stage does use the actual incomplete s40 masks. The fitted residual gain is a *linear proxy for potential additional predictive information*, not a guaranteed gain of a learned nonlinear C2 model. Aggregating many edges introduces multiple-comparison effects: the counts are descriptive exploratory findings. Confirm robustness over thresholds, blocks, different networks and seeds before making publication claims. An official STLA trained on TRAIN has already seen the internal TRAIN holdout; the independent VAL confirmation is more meaningful. This protocol does not establish physical causation.

## Important v1.1 GPU queue hotfix (2026-11-02)

The previous PBS launch used a bare `python` command. Gadi GPU compute jobs do **not** inherit the login shell's active `TFP` conda environment; both previous `PGain` jobs exited immediately with `/bin/python`'s deprecation wrapper and `Exit Status: 1`. **Do not resubmit the v1 PBS file.**

Both PBS files now use the absolute, already-known environment executable:

```
/g/data/mh20/zw1165/envs/TFP/bin/python
```

Override ONLY if that environment has genuinely moved: `-v TFP_PYTHON=/actual/absolute/path,...`. GPU PBS validates CUDA visibility and executes **TRAIN + VAL official STLA preflight** on the allocated node *before* long statistical work. It propagates nonzero Python exits through the logging pipeline.

**Run this command on the login node BEFORE re-queueing** (no GPU required):

```bash
bash PCFT-STLA/diagnostics/propagation_identifiability/scripts/pre_submit_check.sh
```

Only resubmit if it ends with `READY_FOR_GPU_PBS`; this command checks the exact Python binary, imports, shell syntax, unit tests, Speed/Flow checkpoint and cache identities, and one actual TRAIN and VAL inference for each target. It cannot simulate real CUDA, so the PBS compute-node GPU smoke is still required.

The patch additionally handles zero qualified edges, checks TRAIN/VAL tensor dimensions and mask alignment, and avoids overwriting a complete `SUMMARY.json` with CPU stats-only output. An incomplete new full run removes the previous `SUMMARY.json` to prevent mistaking stale output for current results. CPU stats are written as `SUMMARY_STATS_ONLY.json`.

## Installation (unpack at the Speed_LLM_RAG root)

The distributed archive has the path prefix `PCFT-STLA/diagnostics/propagation_identifiability/`. Extract it into your existing root without replacing the existing PCFT model code.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
/g/data/mh20/zw1165/envs/TFP/bin/python -m pytest -q PCFT-STLA/diagnostics/propagation_identifiability/tests
```

### Fast local/Gadi CPU check (no GPU)

```bash
python -u PCFT-STLA/diagnostics/propagation_identifiability/scripts/run_diagnostic.py \
  --phase preflight --target speed \
  --checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt
python -u PCFT-STLA/diagnostics/propagation_identifiability/scripts/run_diagnostic.py \
  --phase preflight --target flow \
  --checkpoint Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt
```

Preflight loads one official STLA checkpoint on CPU, checks physical graph construction, formal cache, and one official STLA inference. It does NOT compute the actual counts.

### CPU-only discovery counts (no STLA forecast), through CPU PBS

```bash
qsub -N PDIR_S -v TARGET=speed PCFT-STLA/diagnostics/propagation_identifiability/pbs/stats_only.pbs
qsub -N PDIR_F -v TARGET=flow  PCFT-STLA/diagnostics/propagation_identifiability/pbs/stats_only.pbs
```

### Complete diagnostic: TRAIN discovery + STLA residual gain + independent VAL confirmation

```bash
qsub -N PGain_S \
  -v TARGET=speed,SEED=10,CHECKPOINT=Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt \
  PCFT-STLA/diagnostics/propagation_identifiability/pbs/propagation_diagnostic.pbs

qsub -N PGain_F \
  -v TARGET=flow,SEED=10,CHECKPOINT=Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt \
  PCFT-STLA/diagnostics/propagation_identifiability/pbs/propagation_diagnostic.pbs
```

Output (CPU-only stats use `SUMMARY_STATS_ONLY.json`):

```
PCFT-STLA_diagnostics/propagation_identifiability/TraffiDent_Net_06/{speed,flow}/
  PREFLIGHT.json                          (CPU preflight only)
  physical_direction_lag_candidates.csv   (all physical directed edges + lag correlations)
  stla_incremental_gain_edges.csv         (eligible stable-direction edges only)
  SUMMARY.json                            (counts, definitions, fingerprints)
PCFT-STLA_logs/propagation_identifiability/*.live.log
```

To share the results (no `.pt`, `.npz`, `.npy`):

```bash
tar --exclude='*.pt' --exclude='*.pth' --exclude='*.npz' --exclude='*.npy' \
  -czf PCFT_PropagationDiagnostic_Results_Small.tar.gz \
  PCFT-STLA_diagnostics/propagation_identifiability/TraffiDent_Net_06 \
  PCFT-STLA_logs/propagation_identifiability
ls -lh PCFT_PropagationDiagnostic_Results_Small.tar.gz
```

## Scope, thresholds, and safeguards

- `--min-abs-corr 0.15`: discovery Pearson magnitude threshold.
- `--min-direction-gap 0.03`: directed lead/lag correlation strength minus reverse strength.
- `--min-sign-fraction 0.75`, `--min-lag-fraction 0.66`: cross-block stability.
- `--min-gain-pct 0.2`: out-of-block / VAL incremental MAE gain threshold.
- `--max-lag 6`: 5–30 minutes. This is a *statistical association*; known wave speeds are not assumed.
- `--confirm-val`: VAL confirmation is enabled by the provided GPU PBS script.
- Exact chronological identity, source hashes, checkpoint SHA256 and cache manifest SHA256 are saved. All writes confined to `PCFT-STLA_diagnostics/` and logs.
- The test scripts never read `test` split or EvaluationCache.

Data and Gadi GPU runtimes are not mounted in this chat; the diagnostic must run in your Gadi project to produce real counts.
