# FARE v4.0.3 — Full Research Implementation

**Traffic Predictive Support Restoration via Directional Evidence Retrieval and Counterfactual Utility**

A **single, full FARE model** with a frozen **STLAformer** forecasting backbone.  Historical candidate selection, graph-aware transport, set-conditional Keep/Swap utility, residual/evolution fusion and risk-scaled final prediction belong to **one computation graph**. There are no independently retrained FARE ablations.

## Research implementation and provenance

- **Real data input:** `Base_SupportBench` Schema v3. No changes to the existing Train/Val/Test, sensing masks, target normalization or canonical `metric_bundle`.
- **Traffic network:** broad non-symmetric adjacency is used for context/regions. The upstream/downstream neighbor arrays determine direction-labeled sparse propagation sources. Input weights are **affinities**, never invented road distances or measured congestion-wave speeds.
- **Memory:** completed historical episodes, target-specific future, residual and 8-dimensional capability signature. Train IDs and actual time windows remain in the cache; historical future must finish **before** query origin.
- **Backbone:** loads the *real* `Backbone/vendor/STLAformer/model/STLformer.py` and the selected frozen checkpoint. The official vendor base weights remain frozen; the evidence-injection modules are trainable. Pre-injection activations are reused across candidate and null predictions.
- **Evidence:** same-node and sparse path-based historical values combined using a learned path gate and bounded nonwrapping temporal alignment. Residual and Evolution VALUE paths do not read each other's values.
- **Utility:** multihead candidate Set Attention + explicitly learned Keep/Swap heads + one legal replacement, budget Top-8 Episodes, two evidence types and Horizon-specific Null.
- **Risk:** one unscaled `candidate_prediction`, one evidence-free `null_prediction`, final `null + alpha*(candidate-null)`, alpha in `[0,1.25]`. A train-only action teacher supervises this *exact same alpha*. Gain and Harm heads receive explicit supervision.
- **Training:** one staged Full model with Train-only observation sensitivity teacher (a *Backbone sensitivity proxy*, not observability ground truth), exact LOO/Swap/Action teacher and replay tied to the frozen teacher candidate sets.
- **Evaluation:** canonical metrics; validation for checkpoint selection; **Test s20/s40/s60/s90 by default**, `--eval-val` if validation also needed.
- **Attribution:** one best checkpoint, independent inference-time interventions; common computations are cached. Expensive inference interventions are budgeted **and explicitly labeled as a sampled diagnostic**, separate from the complete-Test headline metrics. No external baselines are invoked.

### Scientific qualification

Graph hops, weighted direction links and learned lag are **traffic propagation proxies**. The current Net-06 processed files do not verify physical path length or measured congestion-wave velocity. Do not report them as real shockwave ground truth. Do not claim TRC novelty or accuracy improvements until real cluster experiments are completed.

### Dataset status (audited on the user's Gadi environment)

- `TraffiDent_Net_06`, 388 nodes, H=P=12, 50,000 unique Train windows.
- Five balanced failure types: `entry_random`, `regional_outage`, `directional_outage`, `temporal_outage`, `sensor_outage`; Train severity s40.
- Speed and Flow share an identical adjacency and identical upstream/downstream neighbor IDs/weights and Train window/family metadata.
- Net-01/02/03/04 are **not yet processed** in Base_SupportBench. The model is size-dynamic, but those datasets require the same upstream processing and their own STLAformer checkpoints before execution.

## 1. Install without interrupting running v3 jobs

The archive contains the **entire** `FARE/` folder; it does **not** contain `Base_SupportBench`, `Backbone`, raw user data or private Gadi checkpoints. If v3 PBS jobs remain active, **do not overwrite their `FARE/` sources during the run**. After old jobs finish:

```bash
## Run these commands from the Speed_LLM_RAG repository directory:
mv FARE FARE_v3_archived_20261015
unzip FARE_v4.0.3_AMP_Safety_PreSubmit_Full.zip
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/FARE:${PYTHONPATH:-}"
conda activate TFP
```

The resulting directory must be `$SPEED_LLM_RAG_ROOT/FARE`.

## 2. Validate the environment/data BEFORE GPU work

```bash
python -m pytest -q FARE/tests
python -u FARE/scripts/preflight.py --config FARE/configs/pilot.yaml \
  --dataset TraffiDent_Net_06 --target speed --checkpoint "$STLA_SPEED"
python -u FARE/scripts/preflight.py --config FARE/configs/pilot.yaml \
  --dataset TraffiDent_Net_06 --target flow --checkpoint "$STLA_FLOW"
```

`FARE4_PREFLIGHT_PASS` confirms input/checkpoint/direction/schema checks, **not** the result of a real V100 forward.

## 3. Run Full Pilot — one PBS job, one V100 per target

```bash
export STLA_SPEED="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export STLA_FLOW="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"

python -u FARE/scripts/submit_gadi.py --config FARE/configs/pilot.yaml \
  --dataset TraffiDent_Net_06 --target speed --seed 10 --checkpoint "$STLA_SPEED"
python -u FARE/scripts/submit_gadi.py --config FARE/configs/pilot.yaml \
  --dataset TraffiDent_Net_06 --target flow --seed 10 --checkpoint "$STLA_FLOW"
```

The default is a **full Pilot**, not a smoke. No additional `analysis.pbs` job is submitted. Each job requests `gpuvolta`, **1 V100, 12 CPU, 32 GB RAM and 48 hours**. All steps Train → Test Evaluate → Internal Attribution run sequentially inside the **same PBS job and GPU allocation**. This does **not** guarantee completion within 48 hours; actual throughput must be profiled.

Outputs are in `FARE_exps/<target>/TraffiDent_Net_06/pilot/<timestamp>_seed10` and logs in `FARE_logs`.

### Optional smoke for initial real GPU integration

Add `--smoke` to the submit command; the same full architecture is exercised with a small number of windows/epochs. Preflight passing does not prove the vendor source or GPU activation behavior.

## 4. Actual V100 profile (prebuilt Memory required)

```bash
python -u FARE/scripts/build_memory.py --config FARE/configs/pilot.yaml \
  --target speed --checkpoint "$STLA_SPEED"
# Run only inside GPU compute allocation:
python -u FARE/scripts/profile_gpu.py --config FARE/configs/pilot.yaml \
  --target speed --checkpoint "$STLA_SPEED" \
  --memory "$PWD/FARE_cache/speed/TraffiDent_Net_06/memory_v4_10_16000" \
  --steps 20 --out FARE_profile_speed.json
```

`profile_gpu.py` is not a GPU reservation command. Use it **inside a PBS compute job**, not a login node.

## 5. Results and resume

- `best.pt` and `last.pt`: Full model/optimizer/stage/RNG state.
- `train_metrics.jsonl`: per-epoch Train/Val metrics.
- `teacher_*.pt`: exact Train-only Counterfactual snapshots, cached with candidate evidence and labels.
- `support_teacher.pt`: observation-removal sensitivity proxy labels.
- `evaluation/performance_summary.csv`: complete Test s20/s40/s60/s90 + averages; Val if explicitly enabled.
- `analysis/module_contributions.csv`, `research_findings.json`, `research_report.html`: fixed-checkpoint interventions, sampled query counts and uncertainty.

If a job stops before analysis and training is already complete, use `scripts/evaluate.py` / `scripts/analyze.py` with the saved `best.pt` — **no new model training**.

Manual resume from latest stage:

```bash
python -u FARE/scripts/run_pipeline.py --config FARE/configs/pilot.yaml \
  --target speed --checkpoint "$STLA_SPEED" \
  --run-dir "<existing_v4_run_dir>" --resume "<existing_v4_run_dir>/last.pt"
```

The current one-job scheduler requests a new allocation if invoked again; it does not silently allocate a second GPU for analysis.

## 6. Formal OOS residual requirement

**Pilot** uses `checkpoint_in_sample`: exploratory only. **Formal** uses `external_oos` and refuses in-sample Memory.

The full archive contains `scripts/build_oos_predictions.py` to combine externally trained chronological STLAformer fold checkpoints, but it **does not train those fold producers automatically**, because the private `Backbone` project and fold training commands are not in this archive. Formal can run after actual time-purged fold producer checkpoints are supplied. Their last **training label** timestamps must be earlier than the start of the corresponding OOS forecast sample. Verify the producer's training logs independently; an asserted `trained_until` field by itself is not proof of true OOF status.

```bash
python -u FARE/scripts/build_oos_predictions.py \
  --manifest FARE_oos_folds_speed.json --target speed \
  --output FARE_cache/oos_speed.npz
python -u FARE/scripts/submit_gadi.py --config FARE/configs/formal.yaml \
  --target speed --checkpoint "$STLA_SPEED" \
  --oos-forecast-file "$PWD/FARE_cache/oos_speed.npz"
```

## 7. Performance constraints

- Memory stored as random-access `.npy` arrays, not one compressed NPZ; compact Region keys on GPU, node/capability keys and historical values gathered only after Top-K.
- Region Top-K computed in chunks, no `[B,G,M,D]` materialized broadcast.
- Sparse propagation sources and factorized space/time weighting; value branches retain same-node historical evidence.
- Set Attention on `[B,N,K,D]`, not `[B,N,K,P,D]`.
- Frozen STLAformer Prefix shared by base, evidence and LOO/Swap outputs.
- Teacher uses cached candidates and downstream re-execution; periodic limited replay rather than exhausting all training steps.
- Analysis recomputes only the dependency-affected suffix, with a logged query budget for expensive interventions.
- Use FP16 autocast + GradScaler on CUDA; gain/MAE/risk losses use FP32 where necessary.

## 8. Known limitations and important claims

This release has passed **synthetic unit/integration tests** and a 388-node synthetic forward/backward. It has **not been run on the user's actual private Net-06 caches or STLAformer checkpoint on Gadi**. Real smoke/Profiling is still required to establish end-to-end vendor compatibility, throughput, peak V100 memory use, and final MAE/RMSE improvements. It is a complete model/codebase deliverable, not a claim of an empirically validated TRC contribution.

The Train-only support-sensitivity teacher measures the prediction impact of *additional legal observation removal* on the frozen Backbone; it is not a measured physical observability index. Historical residual producer quality and its chronological provenance remain externally auditable scientific prerequisites.


## Path portability / moved Conda environments (v4.0.1)

The repository and Python environment paths are **not hardcoded**. The repo root
comes from the location of `FARE/fare/config.py` by default; setting
`SPEED_LLM_RAG_ROOT` is optional. Input paths are accepted as absolute paths,
paths relative to your current working directory, or paths relative to the
repository. For example, from the repository root after `conda activate TFP`:

```bash
python -u FARE/scripts/preflight.py --target speed
python -u FARE/scripts/submit_gadi.py --target speed --seed 10
python -u FARE/scripts/submit_gadi.py --target flow --seed 10
```

To specify a fixed checkpoint without an absolute path:

```bash
python -u FARE/scripts/submit_gadi.py --target speed \
  --checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/<run>/best.pt
```

`submit_gadi.py` transfers the **currently active** `sys.executable` and
`sys.prefix` into the PBS job. The PBS script uses that exact Python and checks
its prefix, rather than activating `TFP` by name. This avoids accidentally
selecting an old `/scratch/.../envs/TFP` copy when a new environment was moved
under `/g/data`. The transferred paths are absolute **resolved at submission
time**, not hardcoded in FARE source code; PBS requires this to find the same
interpreter on compute nodes. Verify that the active environment is the one
you intend *before* submission.

You can also submit from another working directory using an absolute path to
`FARE/scripts/submit_gadi.py`; the config is resolved against the installed
FARE package automatically. Alternatively, use `--config configs/pilot.yaml`.

`--resume` and `--memory` also accept repository-relative paths and are passed
through to the same PBS job. To resume an earlier v4 run:

```bash
python -u FARE/scripts/submit_gadi.py --target speed \
  --run-dir FARE_exps/speed/TraffiDent_Net_06/pilot/<run> \
  --resume FARE_exps/speed/TraffiDent_Net_06/pilot/<run>/last.pt
```

Use `--evaluate-only` to run post-training evaluation/analysis from a saved
checkpoint without retraining (pass a suitable `--run-dir` and `--resume`), or
`--skip-analysis` to omit the most expensive attribution.

**Legacy checkpoint caveat:** manifests from earlier runs store resolved
checkpoint/memory locations. An unchanged project directory with a moved Conda
environment is fine; moving the *entire workspace* afterward can require
rewriting old saved manifest artifact paths.

## v4.0.2 Teacher sampling hotfix

Fixes the `IndexError` raised by `choices[empty, 0]` during
`generate_teacher()` when the empty-candidate mask is shaped `[B,N]`.
Legal replacements are now sampled through `sample_legal_replacement()`;
rows with no legal unselected candidates receive an index-0 placeholder
with `replacement_valid=False`, which is excluded from Swap supervision.
Adds tests for mixed and all-empty candidate pools. No change to model
architecture, Base_SupportBench cache, or STLAformer checkpoints.


## v4.0.3 FP16 numerical stability and fail-closed pre-submit check

The 2026-10-16 Gadi job `181327616.gadi-pbs` failed on the first GPU training batch because an FP16 retrieval score was masked with `-1e7`, outside the finite FP16 range. The fix covers **all** such masked score/logit paths in Retrieval, Evidence Transport and Set Utility, with production-sized FP16 CPU autocast regressions.

Before queueing a V100, run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q FARE/tests
python -u FARE/scripts/submit_gadi.py --check-only --target speed \
  --checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt
```

`--check-only` runs the CPU FP16 regressions, checks Test Cache s20/s40/s60/s90, strictly loads the **real** vendor STLAformer checkpoint and executes one actual-data CPU STLAformer forward. It then exits **without qsub**. Normal `submit_gadi.py` also runs this gate **by default**, and does not submit if the gate fails. When an actual V100 allocation begins, `training.cuda_preflight: true` also executes a small real STLAformer + FARE CUDA FP16 forward/backward check **before** expensive Memory construction; the test-only evidence is discarded.

See [AMP_SAFETY_4_0_3.md](docs/AMP_SAFETY_4_0_3.md). CPU autocast tests do not guarantee a successful V100 CUDA run.
