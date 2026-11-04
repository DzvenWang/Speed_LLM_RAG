# PCFT-STLA v7.0 — Dedicated CPU Diagnostic v1.0.0

This is a fully integrated **read-only, CPU-only diagnostic** for the trained PCFT v7.0 system. It is distributed **with the original v6 and v7 complete project**. No STLA/PCFT weights, large multi-channel caches or Gadi training output are included in this source package; they remain on the user's Gadi workspace.

It does **not** implement a new forecasting model, alter C1/C2/C3 training, optimize test-set parameters, or identify causal road-wave propagation. The existing v7 `model.py` and `propagation.py` have two **optional diagnostic interfaces**, disabled during training. Default model outputs are unchanged, checked on synthetic tests.

## Diagnostic modes

- `summary`: read existing checkpoint header, `test_results.json`, `epoch_history.jsonl` and `train_partition.json`; **no forward**, no graph/cache access.
- `behavior`: selected or complete Validation forward with C1 missingness, C2 source validity/freshness, C3 candidate/control/gain calibration, H1–H12, masks and time-block bootstrap.
- `intervention`: `behavior` plus **paired, frozen-checkpoint** source ablations and coverage interventions. Reports both policy-recomputed and fixed-original-policy responses; **not** independent retrained ablations.
- `deep`: `intervention` plus per-edge average learned weights on at most 256 windows (default hard bound).

## Gadi install

In existing `/scratch/mh20/zw1165/Speed_LLM_RAG` (contains `Backbone/`, `Base_SupportBench/`, the original best.pt models and existing full PCFT caches):

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
unzip -oq PCFT-STLA_v7.0.0_plus_CPU_Diagnostic_v1.0.0_Complete_Source.zip
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/PCFT-STLA:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
bash PCFT-STLA/diagnostics/v7_cpu/verify_cpu_diag.sh
```

`verify_cpu_diag.sh` runs the existing v6/v7 tests plus all CPU-diagnostic tests. It does not submit GPU jobs or rebuild data.

**Recommended: use a CPU PBS job rather than running full Validation on a shared login node.** The PBS has `normal`, 8 CPU, 32GB memory, 4-hour walltime and `storage=scratch/mh20+gdata/mh20`; adjust the queue if your allocation requires a different CPU queue.

### Speed and Flow CPU pilot (512 windows)

```bash
qsub -N PCFT_CPU_S -v TARGET=speed,MODE=behavior,MAX_WINDOWS=512 \
  PCFT-STLA/diagnostics/v7_cpu/pbs/cpu_diagnostic.pbs
qsub -N PCFT_CPU_F -v TARGET=flow,MODE=behavior,MAX_WINDOWS=512 \
  PCFT-STLA/diagnostics/v7_cpu/pbs/cpu_diagnostic.pbs
```

Default checkpoints (only when unchanged from the original v7 formal runs):

- Speed: `PCFT-STLA_exps/speed/incomplete/TraffiDent_Net_06/v7/formal/20261103_032842_seed10/joint_best.pt`
- Flow: `PCFT-STLA_exps/flow/incomplete/TraffiDent_Net_06/v7/formal/20261103_033807_seed10/joint_best.pt`
- Official STLA Speed: `Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt`
- Official STLA Flow: `Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt`

For another run, explicitly pass `CHECKPOINT` and `BACKBONE` variables (`qsub -v` values should not contain commas in paths). The original checkpoint and all matching v6 multi-channel/graph caches **must exist**. The diagnostic refuses to rebuild missing caches and compares v7 checkpoint identity strictly against current data/graph/backbone identities.

### Full Validation (5,000 windows)

```bash
qsub -N PCFT_CPU_S_FULL -v TARGET=speed,MODE=behavior,MAX_WINDOWS=0 \
  PCFT-STLA/diagnostics/v7_cpu/pbs/cpu_diagnostic.pbs
qsub -N PCFT_CPU_F_FULL -v TARGET=flow,MODE=behavior,MAX_WINDOWS=0 \
  PCFT-STLA/diagnostics/v7_cpu/pbs/cpu_diagnostic.pbs
```

### Pairwise source ablation on 256 windows

```bash
qsub -N PCFT_CPU_S_INT -v TARGET=speed,MODE=intervention,MAX_WINDOWS=256 \
  PCFT-STLA/diagnostics/v7_cpu/pbs/cpu_diagnostic.pbs
qsub -N PCFT_CPU_F_INT -v TARGET=flow,MODE=intervention,MAX_WINDOWS=256 \
  PCFT-STLA/diagnostics/v7_cpu/pbs/cpu_diagnostic.pbs
```

### Deep source trace (<=256 windows)

```bash
qsub -N PCFT_CPU_S_DEEP -v TARGET=speed,MODE=deep,MAX_WINDOWS=128 \
  PCFT-STLA/diagnostics/v7_cpu/pbs/cpu_diagnostic.pbs
```

### Direct CPU command (inside an allocated CPU node)

```bash
export CUDA_VISIBLE_DEVICES=''
/g/data/mh20/zw1165/envs/TFP/bin/python -u PCFT-STLA/diagnostics/v7_cpu/run.py \
  --target speed --dataset TraffiDent_Net_06 --split val \
  --checkpoint PCFT-STLA_exps/speed/incomplete/TraffiDent_Net_06/v7/formal/20261103_032842_seed10/joint_best.pt \
  --backbone-checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt \
  --mode behavior --max-windows 512 --batch-size 8 --cpu-threads 8 \
  --output PCFT-STLA_diagnostics/v7_cpu/manual_speed_512
```

`--max-windows 0` means all windows, except forbidden in `deep` mode. `--interventions` can specify any of: `no_functional no_physical forward_only reverse_only fresh_only skip_coverage_multiply no_c1 no_c2 no_local no_direction no_null`. By default, intervention/deep modes perform the first six source/coverage tests. `--max-fresh-age 3` sets the freshness intervention to at most three 5-minute steps old. In cases where a required source is unavailable, all missing nodes stay unavailable: no invented sensor values.

Test-set analysis is disabled unless `--allow-test` is explicitly passed; all new model decisions should be made on Validation, and test reports should remain descriptive.

## Outputs

Each CPU PBS job creates a **separate timestamped diagnostics directory**, not an experiment or cache directory. Files include:

| File | Interpretation |
|---|---|
| `summary.json` | Raw-unit MAE/RMSE, original STLA, Local, Propagation, Controlled, fixed-shrink, uniform, hard action, diagnostic discrete oracle; percentage improvement relative to STLA |
| `run_manifest.json`, `sample_manifest.csv` | Checkpoint/graph/backbone identities, sample IDs, seed and auditable config |
| `c1_support_groups.csv` | Target coverage, latest observation age, max missing streak, valid historical difference count, exact group MAE |
| `c2_source_groups.csv` / `c2_message_statistics.csv` | Per-target number of valid incoming **edges**, source freshness, null source, source coverage and effect by group; overlapping edges are counted as edges, not unique sensors |
| `c3_action_statistics.csv`, `c3_gain_calibration.csv` | Actual action-weight entropy/argmax, normalized predicted versus actual Gain, predicted versus actual harmful frequencies |
| `intervention_results.csv` | Matched frozen model source/coverage perturbations; both recomputed and frozen original policy; no new training |
| `horizon_results.csv`, `mask_results.csv` | Stepwise H1–H12 and direct-mask group metrics |
| `window_metrics.csv`, `bootstrap_intervals.json` | Weighted per-window MAE, day-sized time-block bootstrap of improvement; small samples may not permit CI |
| `selected_cases.csv` | Highest absolute window-level improvement/retreat for manual review |
| `edge_weight_summary.csv` | Deep mode only: mean learned per-edge attention across sampled windows and horizons; **not** source causal effect |
| `report.md` | Compact self-contained report and interpretation limitations |

The discrete oracle chooses the lowest-MAE candidate separately at each forecast position *using the future truth*, so it is **not deployable**, and learned soft mixtures can be better than the discrete oracle. Improvement under post-hoc interventions is not equivalent to benefit from retraining such a model.

To upload diagnostics for detailed analysis, in the project root:

```bash
bash PCFT-STLA/diagnostics/v7_cpu/package_results.sh
```

This creates `PCFT_v7_CPU_Diagnostic_Analysis_Small.tar.gz`, excluding all `.pt`, `.npz`, `.npy` and `.parquet` files. It is safe to upload that small text archive.

## Engineering guarantees and limitations

1. Never passes `y_norm`, `y_raw`, `future_valid` into the forecasting `model()` as causal features. These are read *after* prediction only for statistical scoring.
2. `torch.inference_mode()`, `.eval()`, CPU only. No optimizer, training, backward or saving model tensors. Existing output/cache locations are rejected as writable diagnostic results.
3. Before `prepare()`, all original v6 multi-channel caches and the graph with the correct source fingerprint are verified to exist and match; the tool refuses to materialize missing caches.
4. `return_diagnostics=False` is the original v7 computation. Unit tests compare exact default forecasts against trace-enabled forecasts.
5. Fully missing sources produce no propagation correction; CPU test cases cover this and source masking.
6. `summary` does not load STLA or data. `behavior`, `intervention`, `deep` all use real checkpoints and data when run on Gadi.
7. Bootstrap resamples blocks based on original `window_start // 288`; output is conditional on the selected fixed checkpoints and source windows. When fewer than five chronological blocks exist, no CI is claimed. Distinct `window_start` indices are not assumed independent individual observations.
8. Project contains no weights or external `Backbone`/`Base_SupportBench` source. The user's existing Gadi installation provides those. No actual Gadi CPU PBS jobs have been run as part of this packaged-source verification.

**Publication target:** Speed and Flow Overall MAE each >5% better than the *same-protocol* strong STLA baseline, with independent network/seed verification. CPU diagnostics support design decisions but cannot establish an expected improvement before retraining.
