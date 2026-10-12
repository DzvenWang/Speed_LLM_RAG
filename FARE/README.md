**Current development version: FARE v2.0.3.** See [README_PILOT_V203.md](README_PILOT_V203.md) and [RELEASE_2.0.3.md](RELEASE_2.0.3.md).

# Current release: v2.0.2 (resources + online diagnostics). See README_V202_START_HERE.md and RELEASE_2.0.2.md.

# FARE v2.0.1 performance release

Start with [README_V201_START_HERE.md](README_V201_START_HERE.md) and [RELEASE_2.0.1.md](RELEASE_2.0.1.md).

> **v1.3.2 Same-Allocation Gadi Pilot & Formal Evaluate (2026-10-08):**
> A normal `submit_gadi.py train` now submits **one PBS job per target**.
> The selected `candidate_best.pt` is evaluated immediately after training,
> using the **same GPU allocation**, without a separate held Evaluate job.
> Pilot defaults to Validation; Formal defaults to s20/s40/s60/s90 Test.
> Explicit `--eval-splits val test` enables exploratory Pilot Test **after**
> Validation checkpoint selection. See `RELEASE_1.3.2.md`.
> Manual `submit_gadi.py evaluate` remains available. The explicitly requested
> `--finalize` OOF/calibration chain still creates additional dependent jobs.
>
> **v1.3.1 Gradient Audit Hotfix (2026-10-08):** `scripts/diagnose_gradients.py`
> initializes the Train-only retrieval key index with `m.refresh(mem)` before
> the first forward pass and rebuilds it after each optimizer update. The audit
> now reports `FARE_GRADIENT_INDEX_READY`, rejects nonfinite parameter gradients,
> and fails if fewer than the requested steps run. Model architecture and training
> hyperparameters are unchanged; **v1.3.0 checkpoints remain compatible**.
> See `HOTFIX_1.3.1.md`.

> **v1.1.0:** Full upgrade, not a small patch. Bounded, learned correction amplitude; candidate forecast supervision; calibration that matches actual inference; OOF alpha consistency; checkpoint-exact evaluation; validation alpha diagnostic. Read `RELEASE_1.1.0.md`. **Start a new run: v1.0.4 checkpoints are not compatible.**

> **v1.0.4 Smoke Retrieval Coverage:** Smoke now spreads its memory across TRAIN history, preserving legal overlap exclusion. Epoch logs include TRAIN/Validation retrieval validity, candidate delta, and gate strength; zero valid TRAIN retrieval makes smoke fail. See `HOTFIX_1.0.4.md`.

> **v1.0.3 Gadi Auto/Manual Evaluate:** Gadi formal training defaults to post-train s20/s40/s60/s90 testing. Historically `scripts/submit_gadi.py train` queued independent `afterok` Evaluate jobs; v1.3.2 changes the normal workflow to same-allocation evaluation. Manual `scripts/submit_gadi.py evaluate` can target candidate/best/phase checkpoints. See `HOTFIX_1.0.3.md`.
> **v1.0.2 joint smoke runner:** Single command checks both speed and flow (preflight, all five training phases, per-phase validation, checkpoints) and writes a combined pass/fail report. See `HOTFIX_1.0.2.md`.
> **v1.0.1 stability hotfix:** Fixes zero-variance NaN gradients and masked loss accounting. See `HOTFIX_1.0.1.md`.

# FARE v1.1.0 Full — Forecast-Calibrated Retrieval Integration

**Research implementation** (initial engineering release, not a validated accuracy claim).

## Status and dependencies

This package contains actual PyTorch implementations of the FARE prediction model, train-only memory, graph target groups, directed-anchor observability, two-stage retrieval, state recovery, cross-attention-based innovation recovery, modified STLAformer, risk/utility controller, phases, stage evaluation, severity evaluation, OOF fitting, calibration, ablations and Gadi PBS wrappers.

FARE **does not import STLA_DASR, Support_Oracle, the old FARE project, Mask_RAFT-STLA, Mask_RAST-STLA, or Baseline**. It imports only `Backbone` STLAformer code/weights and the frozen Base_SupportBench contract/metrics.

The supplied Backbone ZIP excludes its third-party `vendor/STLAformer/model/STLformer.py`; install/preserve the original Backbone vendor source locally/Gadi before running. FARE will explicitly fail preflight when the vendor/checkpoint is absent. The test suite uses a *synthetic STLA-compatible fixture*, not the actual official model or traffic dataset. No real-world forecasting accuracy is claimed or simulated.

Dependencies: Python 3.11+, PyTorch, NumPy, pandas, PyYAML, and an installed local `Base_SupportBench`, `Backbone`, `Backbone_exps`, frozen `ExperimentCache` and their resolved SupportBench configs. You can use the current `TFP` environment. Graph directions are loaded from each dataset's `data/processed/traffident/base_supportbench/{target}/{dataset}/graph/directed_neighbors.npz`.

## Installation

Extract the ZIP so its top-level `FARE/` folder sits beside `Backbone/` and `Base_SupportBench/`:

```text
Speed_LLM_RAG/
├── Base_SupportBench/
├── Backbone/
├── Backbone_exps/
├── Base_SupportBench_artifacts/
├── ExperimentCache/
└── FARE/
```

Windows PowerShell:

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path
$env:PYTHONUNBUFFERED = '1'
python -u FARE\scripts\smoke_all.py --dataset TraffiDent_Net_06 --seed 10
# For a single target, use preflight.py and train.py --smoke as before.
```

Net-01, Net-02, Net-03, Net-04 and Net-06 share the **same source and hyperparameter template**; pretrained STLAformer weights, directed graph, nodes, normalization, retrieval store and calibration are per-dataset and per-target. To switch networks change only `--dataset`.

## Combined smoke (Speed + Flow)

```powershell
python -u FARE\scripts\smoke_all.py --dataset TraffiDent_Net_06 --seed 10
```

The targets run **sequentially** on a single GPU; both are attempted and success is reported only if both pass preflight, all five training phases, and phase validation artifacts. Each target writes separate logs, checkpoints, and phase metrics. The combined report is at `FARE_logs/smoke/<dataset>/<timestamp>_seed10/smoke_summary.json`. Exit code is nonzero if either target fails. Net-01/02/03/04/06 use the same runner; change only `--dataset`.

The combined smoke does **not** include expensive OOF utility fitting or formal s20/s40/s60/s90 test evaluation. Those remain independent formal pipeline tasks.

## Full formal workflow

1. Check the immutable data contract and the correct STLA checkpoint:

```powershell
python -u FARE\scripts\preflight.py --dataset TraffiDent_Net_06 --target speed
```

2. Build a Train-only historical memory artifact (optional, if not present `train.py` builds one and caches latent tokens):

```powershell
python -u FARE\scripts\build_memory.py --dataset TraffiDent_Net_06 --target speed
```

3. Train FARE with phase-specific checkpoints and validation metrics:

```powershell
python -u FARE\scripts\train.py --dataset TraffiDent_Net_06 --target speed --seed 10
```

The training phases are `retrieval`, `recovery`, `injection`, `partial`, and `full`. Each phase has its own `*_best.pt` and `stages/{phase}/val_summary.json` plus `val_retrieval.npz` showing candidate IDs and scores. `history.jsonl` contains every epoch and loss component. The selected candidate is `candidate_best.pt`.

4. Fit the **out-of-fold risk/utility controller** from independently trained, temporally purged fold models. This step is computationally expensive: each OOF fold really retrains its own candidate; it does *not* use the main model's in-sample predictions as OOF supervision.

```powershell
python -u FARE\scripts\build_oof.py --checkpoint "<run_dir>\candidate_best.pt"
```

Creates `oof/oof_labels.npz`, `oof/oof_manifest.json`, and `controller_best.pt`.

5. Calibrate on the frozen Validation split and save `best.pt`:

```powershell
python -u FARE\scripts\calibrate.py --checkpoint "<run_dir>\controller_best.pt"
```

6. Run independent test evaluation:

```powershell
python -u FARE\scripts\evaluate.py --checkpoint "<run_dir>\best.pt" --split test --severities 20 40 60 90
```

Outputs `evaluation/s20`, `evaluation/s40`, `evaluation/s60`, `evaluation/s90` with predictions, Base/Candidate/Final metrics, family metrics, candidate IDs/weights, retrieval quality, state recovery, helpful/harmful corrections and gates.

**One-command sequential pipeline (long running):**

```powershell
python -u FARE\scripts\run_pipeline.py --dataset TraffiDent_Net_06 --target speed --seed 10
```

On Gadi split into PBS steps instead, because the OOF folds and joint training are not guaranteed to fit a single 48h allocation.

## Stage evaluation and ablations

```powershell
python -u FARE\scripts\stage_evaluate.py --run-dir "<run_dir>" --phase retrieval --split val
python -u FARE\scripts\stage_evaluate.py --run-dir "<run_dir>" --phase recovery --split val
python -u FARE\scripts\stage_evaluate.py --run-dir "<run_dir>" --phase injection --split val
python -u FARE\scripts\stage_evaluate.py --run-dir "<run_dir>" --phase partial --split val
python -u FARE\scripts\stage_evaluate.py --run-dir "<run_dir>" --phase full --split val
python -u FARE\scripts\ablate.py --checkpoint "<run_dir>\best.pt" --split val
```

Available inference ablations: `full`, `retrieval_off`, `global_retrieval` (global coarse retrieval; target re-ranking remains), `no_mask_alignment`, `random_memory`, `state_only`, `innovation_only`, `no_risk`. These are **inference ablations**, not separately retrained ablations; use separate training configs/seeds for publishable retrained ablations. Since no-retrieval and full models share network weights, comparing both does not confound two unrelated pretrained Backbones.

To recover an interrupted phase, use the same run directory and the latest checkpoint:

```powershell
python -u FARE\scripts\train.py --dataset TraffiDent_Net_06 --target speed --run-dir "<run_dir>" --resume "<run_dir>\last.pt"
```

A `resume` restores model and optimizer state and resumes at the saved phase/epoch. If the phase is already in `phase_summary.json`, it resumes with the next phase after loading its best phase weights.

## Gadi: one PBS allocation for Training + Candidate Evaluate (v1.3.2)

Run the **submission commands on the Gadi login node**, from the repository root:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
export SPEED_LLM_RAG_ROOT="$PWD"
conda activate TFP

# Submit two Train PBS jobs total (Speed/Flow).
# Each job evaluates candidate_best.pt in the SAME GPU allocation after Train.
python -u FARE/scripts/submit_gadi.py train \
  --dataset TraffiDent_Net_06 --targets speed flow --seed 10
```

The default post-train Evaluate tests **`candidate_best.pt`**, the best
forecast candidate after the five training phases. Its canonical output is
`<run_dir>/evaluation_candidate/` (including `s20/`, `s40/`, `s60/`, `s90/`).
This checkpoint is **not** the final OOF-calibrated `best.pt`.

For the **entire final FARE pipeline**, including OOF-trained controller and
Validation calibration, use `--finalize`. This submits a second dependent
chain `train+candidate-Evaluate -> OOF -> calibration -> best.pt Evaluate`.
**This explicitly requested finalization is not a single-job workflow.**

```bash
python -u FARE/scripts/submit_gadi.py train \
  --dataset TraffiDent_Net_06 --targets speed flow --seed 10 --finalize
```

The final calibrated test outputs `evaluation/s20/` through `evaluation/s90/`.
This pipeline is computationally expensive and its OOF phase may require
additional resources or adjusted walltime based on real-run profiling.

You can also use raw qsub commands. Direct training qsub enables
same-allocation post-training candidate Evaluate by default. The same 48h
walltime covers **both Train and Evaluate**:

```bash
qsub -N FARE06sp -v DATASET=TraffiDent_Net_06,TARGET=speed,SEED=10 FARE/pbs/train_formal.pbs
qsub -N FARE06fl -v DATASET=TraffiDent_Net_06,TARGET=flow,SEED=10 FARE/pbs/train_formal.pbs
```

For **manual test Evaluate** after a checkpoint exists:

```bash
# Final calibrated model (requires `best.pt` from calibration):
python -u FARE/scripts/submit_gadi.py evaluate \
  --run-dir /scratch/mh20/zw1165/Speed_LLM_RAG/FARE_exps/speed/TraffiDent_Net_06/formal/<run_id> \
  --checkpoint-kind best

# Post-train candidate model before OOF calibration:
python -u FARE/scripts/submit_gadi.py evaluate \
  --checkpoint /absolute/path/to/candidate_best.pt --checkpoint-kind candidate

# Independently evaluate an earlier training stage on Validation:
python -u FARE/scripts/submit_gadi.py evaluate \
  --run-dir /absolute/path/to/run_dir --checkpoint-kind recovery --split val

# Evaluate chosen severities and an ablation without overwriting auto output:
python -u FARE/scripts/submit_gadi.py evaluate \
  --checkpoint /absolute/path/to/best.pt --mode retrieval_off \
  --severities 40 60 90
```

The manual PBS Evaluate writes to `<run_dir>/manual_evaluation/` in a new
run-specific directory unless `--output` is explicitly supplied. Manual jobs
can alternatively be submitted directly using `qsub -v CHECKPOINT=... FARE/pbs/evaluate.pbs`.
Do not use the `best` checkpoint kind unless calibrated `best.pt` actually exists.

Job IDs are saved in `<run_dir>/gadi_jobs.json`; live logs are under
`FARE_logs/<target>/<dataset>/` (training/OOF/calibration) and
`FARE_logs/evaluate/<target>/<dataset>/` (evaluation). `FARE_GADI_EVALUATE_DONE`
is printed only after Evaluate exits successfully. Change only `--dataset`
for Net-01/02/03/04/06; each target uses distinct memory and checkpoint.
PBS project/queue/storage directives can be adjusted for your Gadi allocation.

## Scientific and technical contracts

- A **single architecture** runs on Net-01/02/03/04/06; no hardcoded `N=388` or Net-06 node IDs.
- Only valid Train-window observations form the retrieval memory. Exact/overlapping history+forecast windows are excluded by `window_start`.
- Candidate comparison uses the **intersection** of observed Query anchors and originally valid historical candidate anchors.
- Retrieval uses two stages (coarse normalized region keys, then differentiable masked state/trend re-ranking).
- State Recovery applies candidate-weighted historical states plus a graph-walk-propagated observable-anchor difference, with uncertainty and preservation of known observations.
- Innovation Student cross-attends to retrieved historical node-level latent tokens. Teacher difference `full - recovered` is available **only inside training loss**, never inference.
- The STLAformer wrapper preserves official weights and adds recovery embeddings, decomposition adapters, graph-aware reliability adapters and actual Top-K cross-attention in selected STLA blocks.
- Base and retrieved predictions use **the same STLAformer parameters**. Final control is a calibrated mixture; no retrieval candidate means exact base fallback.
- `x_full` and `y_raw` are not arguments to `FARE.forward()`.
- Canonical MAE/RMSE and directly masked MAE/RMSE are imported from Base_SupportBench.
- `best.pt`, serialized Train Memory, graph partition buffers, Backbone identity/hash, stage history and calibration form the evaluation identity.

## Practical constraints and known validation gaps

1. **Only the synthetic small-graph path is tested inside this delivered ZIP**. Actual Net-06 and Net-01..04 runs, official vendor STLAformer numerical equivalence, V100 fit and predictive gains require your connected project runtime. All corresponding scripts exist, but formal execution is not claimed.
2. The full `FARE` uses approximate graph-based anchor transfer, rather than a proven physical traffic-flow model. Compare graph/no-graph in further experiments.
3. Retrieval index keys are recomputed from model weights at stage/evaluation boundaries; they are not mixed from unrelated checkpoints. Coarse hard Top-K is discrete and is not differentiable; Query/Key receive paired-representation supervision while re-ranking receives differentiable weighting.
4. OOF fitting is truly fold-specific but expensive; start with Net-06 and smaller **legitimately isolated** development cache before scheduling all 10 formal runs.
5. The risk controller estimates utility/risk and calibrates on Validation; it does **not** mathematically guarantee RMSE improvement.
6. For formal claims, add multiple seeds, separately retrained ablations, parameter/latency benchmarks, and leakage audits of every data asset.

## Tests

```bash
python -m compileall -q FARE
python FARE/tests/test_full.py
python FARE/tests/test_smoke_all.py
python FARE/tests/test_smoke_all_integration.py
python FARE/tests/test_gadi_submit.py
python FARE/tests/test_gadi_pbs_eval.py
```

Tests use synthetic model weights/data and verify forward, gradients, no candidate fallback, inference label isolation, checkpoint round trip and stage-specific training + evaluation artifact writing. They are functional tests, **not** traffic prediction accuracy benchmarks.

## v1.1.0 forecast-correction diagnostics and configuration

The retrieval-supported candidate is a **proposal**, not an `alpha=1` forecast.
The model learns a bounded `alpha` (default 0..0.60; prior 0.12) from
retrieval confidence, uncertainty, predicted utility/risk, candidate delta,
node and horizon features, then outputs `base + alpha * (candidate - base)`.
The effective gate saved in predictions is exactly this alpha, **including**
Validation multiplier calibration. No hardcoded network IDs or speed/flow
smoke-derived alpha values are used.

After FULL validation Evaluate (no `--smoke`):

```powershell
python -u FARE\scripts\evaluate.py --checkpoint "<run_dir>\candidate_best.pt" --split val --output "<run_dir>\val_diagnostic"
python -u FARE\scripts\alpha_sweep.py --predictions "<run_dir>\val_diagnostic\val_full_predictions.npz"
```

This writes a scalar correction `alpha_sweep.csv` and `alpha_sweep.json`.
This scan is **for validation diagnosis only**, does not overwrite a checkpoint,
and cannot infer node-specific efficacy or extrapolate from a tiny smoke val.
Validation `calibrate.py` fits a multiplier for the learned dynamic gate,
which is capped to `control.max_alpha`.

The original 1.0.4 `candidate_best.pt` cannot be loaded into 1.1.0 because
controller weights and buffers changed. Existing results should be retained
as an experimental baseline rather than overwriting old checkpoints.

A custom full YAML may be submitted on Gadi with
`python -u FARE/scripts/submit_gadi.py train --config /path/to/full.yaml ...`.
Evaluate and OOF always rebuild architecture from the saved checkpoint config.
Recommended complete tests: `python FARE/tests/test_fusion_v110.py`, plus all
existing tests listed above.

---

## v1.2.0: target restoration-supervised Retrieval and corrected ablations

Read [`RELEASE_1.2.0.md`](RELEASE_1.2.0.md) for rationale, safety conditions, commands, and experimental cautions. In short: the previous `state_only` and `no_risk` interventions were not strict. This release fixes them, adds leak-safe TRAIN-only hidden-state restoration ranking and a separately retrained `no_memory` FARE-STLA control. The method supports five networks and speed/flow without any hardcoded Net-06 node count.

New CLI:

- `python -u FARE/scripts/train.py --dataset TraffiDent_Net_06 --target flow --seed 10 --variant no_memory --smoke`
- `python -u FARE/scripts/ablate.py --checkpoint /path/to/candidate_best.pt --split val` — per-mode CSV/JSON and prediction artifacts.
- `python -u FARE/scripts/report_ranking.py --run-dir /path/to/run_dir` — candidate ranking proxy diagnostics.
- `python -u FARE/scripts/submit_gadi.py train --dataset TraffiDent_Net_06 --targets speed flow --variant no_memory` — runs candidate Evaluate within the same Train PBS allocation for the retrained no-memory comparator.

After updating source, do not `--resume` old v1.1.0 optimizer/checkpoint states into the v1.2.0 training pipeline. Run new v1.2.0 training for fair comparative results.

---

## v1.3.0 changes (2026-10-08)

See `RELEASE_1.3.0.md` for the full upgrade and executable commands. Highlights:

- Normalized Ranking metrics on the same target-label-eligible support, with `ranking_target_valid_weight_mass`.
- Actual Observability embedding use in retrieval Query, Recovery gate, and Reliability/Controller pathway.
- `scripts/diagnose_gradients.py` and `scripts/diagnose_fixed_pool.py` for real-cache gradient and fixed-candidate-pool checks.
- `configs/pilot.yaml`: 8k time-spread train records, 16k legal train-only memory entries, 5k validation; never pick alpha or modules on Test.
- Existing five stages and Gadi Train→Auto Evaluate / Manual Evaluate are preserved.
- **Not backward checkpoint-compatible with v1.2.0**; re-train v1.3.0.


## Gadi PBS startup fix (v1.3.3)

Gadi job output `**/etc/bashrc: line 12: BASHRCSOURCED: unbound variable**` means the PBS shell exited **before** Python/CUDA training began; it is not a model or memory error. Starting in v1.3.3 all four PBS entrypoints (`train_formal`, `evaluate`, `oof_controller`, `calibrate`) source only `FARE/pbs/_activate_tfp.sh`, which **never sources `~/.bashrc` or `/etc/bashrc`**. It finds `conda.sh` in the parent directory of `Speed_LLM_RAG` (the Gadi layout has `../miniconda/etc/profile.d/conda.sh`) or uses the optional `FARE_CONDA_SH` override. During `conda.sh` loading and activation, nounset is temporarily disabled and then restored. The script exits with an actionable `FARE_CONDA_INIT_ERROR` or `FARE_CONDA_ACTIVATE_ERROR` on failure.

Verify the bootstrapping from the Gadi login node without booking a GPU:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
export SPEED_LLM_RAG_ROOT="$PWD"
bash --noprofile --norc -c 'set -euo pipefail; source FARE/pbs/_activate_tfp.sh; python -c "import sys; print(sys.executable)"'
```

Expected: `FARE_CONDA_INIT_OK`, `FARE_CONDA_READY ... env=TFP`, and the correct Python executable. If auto-discovery fails, set `export FARE_CONDA_SH=/scratch/mh20/zw1165/miniconda/etc/profile.d/conda.sh` in the login environment to test it; `submit_gadi.py` will forward it in the PBS `-v` environment, including manual Evaluate and optional finalization jobs.

**Already-submitted failed PBS jobs cannot be restarted by editing source files.** After installing v1.3.3, submit a fresh Pilot. This still creates exactly **one Train + in-allocation Val/Test Evaluate PBS job per target**. Train and Evaluate share the same GPU allocation and walltime. No model architecture or checkpoint format changes.


## v1.3.4 — P0.1 evidence-attribution diagnostic (single PBS job)

See [RELEASE_1.3.4_P01.md](RELEASE_1.3.4_P01.md). The **complete unchanged
training pipeline** is retained. P0.1 is a frozen-checkpoint, validation-only
research diagnostic run sequentially for Speed and Flow within one Gadi V100
allocation. It validates s40 mask parity and never changes existing weights.

```bash
qsub -N FARE_P01 -v SEED=10,PILOT_STAMP=20261009_140229_322315_seed10 FARE/pbs/p01_diagnostic.pbs
```
