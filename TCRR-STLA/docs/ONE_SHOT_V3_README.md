# TCRR / PCFT v3.0 – independent One-Shot diagnostics (research source)

Both packages include the complete **unchanged trained model family** plus new `scripts/oneshot_v3.py`, `diagnostics_v3/common.py`, `scripts/compare_oneshot_protocol.py` and `pbs/oneshot_v3.pbs`.

## Intended experiment

- **Two PBS jobs total:** one TCRR, one PCFT; each executes the 2026-10-21 **Speed M3 + Flow M3** saved checkpoints sequentially. No training or GPU smoke test.
- `MAX_QUERIES=0` (default) means **all 5,000 queries per severity**, all s20/s40/s60/s90, both targets; no downsampling. Full forecast horizon H1–H12.
- Reuses the original `Backbone`, `Base_SupportBench`, the immutable TCRR Teacher, the PCFT saved `best_memory_index.npz` and Train-only memories.
- Does not alter the saved checkpoint or model optimizer. PCFT source has an explicit *inference-only* `no_support_anchor` mode; no model weights or original forward in Full mode are changed.
- Original training/test data and checkpoints remain on Gadi, **not** included in the ZIP.

## Gadi installation

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
unzip -o TCRR_PCFT_v3_OneShot_Full_Projects.zip
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
qsub -N TCRR3_DIA TCRR-STLA/pbs/oneshot_v3.pbs
qsub -N PCFT3_DIA PCFT-STLA/pbs/oneshot_v3.pbs
```

The PBS scripts already select these exact runs, and fail if missing:

| Model | Speed | Flow |
|---|---|---|
| TCRR | `20261021_160134_seed10_full_M3` | `20261021_201318_seed10_full_M3` |
| PCFT | `20261021_202703_seed10_full_M3` | `20261021_204016_seed10_full_M3` |

Optional explicit paths: `qsub -v SPEED_CHECKPOINT=/path/best.pt,FLOW_CHECKPOINT=/path/best.pt ...`.
The scripts allow `BATCH_SIZE=...` and `MAX_QUERIES=...` for a targeted run, but the defaults **do not sample**. PBS queue: gpuvolta; ncpus=12; one V100; mem=32 GB; max walltime=48h. Test-time AMP follows saved checkpoint flags; no mixed-precision training.

### Resume

Results are committed after each entire severity; a retry skips finished matching checkpoint+sample-count+mode work. No per-batch resume (a partial severity reruns). Use `qsub -v FORCE=1 ...` to regenerate all outputs. Do not modify model code or checkpoints during execution.

### Where output goes

```text
TCRR-STLA_exps/diagnostics_v3/TraffiDent_Net_06/seed10/{speed,flow}/
PCFT-STLA_exps/diagnostics_v3/TraffiDent_Net_06/seed10/{speed,flow}/
```

Each `s20/s40/s60/s90` has:

- `manifest.json`: checkpoint and Backbone SHA256, input protocol hashes, base forecast SHA256, modes, full experiment reproduction check, and limitation labels;
- `protocol_queries.csv`: exact Query ID, Window Start, per-field hashes (`x`, Mask, Calendar, Future Label, Valid and Direct Mask, raw Base prediction);
- `query_modes.csv`, `query_horizon.csv`, `mode_summary.csv`, `family_summary.csv`, `family_horizon.csv`;
- `mechanism_queries.csv`: method-specific C1/C2/C3 signals and offline gains.

The per-target consolidated reports include `mode_severity.csv`, `mode_family.csv`, `mode_family_horizon.csv`, `mode_average.csv`, `mechanism_queries.csv`, `paired_block_bootstrap.csv` (288 five-minute ticks per temporal block) and `summary.json`.

TCRR also outputs `cf_reference_train_only.csv` and `.json` based on 256 TRAIN queries by default. The complete train histories are used **only for those supervised mechanism labels**. Test full histories are never requested or passed to the model. The reference is not an unbiased held-out metric; it cannot prove Test-time CF recovery accuracy.

## Offline protocol comparison

After both PBS jobs complete, on the **login node without GPU**:

```bash
python TCRR-STLA/scripts/compare_oneshot_protocol.py --root "$PWD"
```

This writes `TCRR_PCFT_v3_protocol_comparison.json`. It reports mismatching fields and first mismatching Query indices. Exit status 1 on any input, target, Backbone or Base-Forecast SHA256 mismatch. The precise Base-Forecast SHA comparison is intentionally strict; minor batch-related floating-point differences may be investigated separately. Do not claim inter-method comparability before checking it.

## Mechanisms and interpretation

### TCRR C1

- Full; No Normal Residual; No Failure Residual; shuffled historical CF pairing; C1 signal magnitudes.
- Pairing shuffle keeps the same candidate pool but perturbs failure-to-normal pairing. It is a same-checkpoint mechanism intervention, **not independently trained**.
- Train-only Teacher CF reference measures normalization-space response-recovery MAE; this is not Test CF truth.

### TCRR C2

- Full vs `no_support_conditioning` re-executes downstream candidate ranking/fusion, keeping retrieved candidates fixed.
- `similarity_only_frozen_gate`: same candidates, same learned None logits, only fixed historic similarity candidate ranking, **Full gate held fixed**, so a *proxy test*, not independently trained baseline or a full isolated C2 effect.
- Single-candidate hindsight Oracle, top learned candidate gain and gain weighted by learned candidate weights; all oracles use labels **only after** inference.

### TCRR C3

- No Gain/Harm vs Full, No Risk vs Full; estimated gain/harm/uncertainty, realized positive/negative contributions, none weight and gate distribution.

### PCFT C1

- `no_support_anchor` is a true forward-path intervention that bypasses the v3 Support Anchor transformer; neither new checkpoint nor alternate model is trained.

### PCFT C2

- `no_lag_alignment`, `no_directional_graph`, `observed_support_only`: actual re-forward through the changed propagation pathway, preserving action selection. Graph is **statistical**, not proof of causal physical propagation.
- Direct full-amplitude Legacy vs Propagation forecasts; valid Episode count, raw Propagation magnitude and availability; results for all five failure types and horizons.
- No forcing of Propagation selection: direct actions are **diagnostic counterfactuals**, not deployable full-policy predictions.

### PCFT C3

- Exact action-only reconstruction of Legacy-only/Propagation-only/No Utility/No Risk based on one Full Forward, checked against model's v3 policy equations. None remains selectable in policy-only modes.
- Predicted vs actual action gains/harm, selected weights, hindsight best action, negative transfer and forecast-support uncertainty.

## Very important limitations

1. **Frozen M3 inference interventions do NOT replace independent M0/M1/M2 training.** This release diagnoses mechanisms, not claims causal independent performance gains of C1/C2/C3.
2. The post-hoc oracles use the Test future only to score hypothetical actions, never as a model input, retrieval feature or learned threshold. Hindsight Oracle gains are not deployable expected improvements.
3. The original per-severity Full evaluation output is read **only to verify numerical reproduction**. Diagnostics do not overwrite original `evaluation/`, model checkpoint, retrieval audit, or memory index.
4. Source contains test-only augmentations and writes diagnostic summaries; no user dataset files or Python external package binaries are shipped.
5. V100 GPU workloads are not executable in this delivery environment; unit/integration tests and Python/PBS syntax have been checked locally. Actual four-checkpoint execution on Gadi remains to be verified by the generated manifests and `*_V3_ONESHOT_PASS` messages.

## Package-only results export

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
zip -r -9 TCRR_PCFT_v3_OneShot_Results.zip \
  TCRR-STLA_exps/diagnostics_v3 PCFT-STLA_exps/diagnostics_v3 \
  TCRR-STLA_logs/diagnostics_v3 PCFT-STLA_logs/diagnostics_v3 \
  TCRR_PCFT_v3_protocol_comparison.json \
  -x '*.pt' '*.pth' '*.ckpt' '*.npy' '*.npz' '*.pyc' '*/__pycache__/*'
```

No re-training and no existing result cleanup is needed.
