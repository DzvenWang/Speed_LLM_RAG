# PCFT-STLA v6.0.0 — Audited Physical–Functional Propagation Forecasting

**This is a complete independent PCFT v6 Python project**, derived from a reviewed v5.0.3 source snapshot but **not reading any old PCFT run result, teacher cache, learned graph, or checkpoint**. It intentionally uses only the existing, read-only sibling `Backbone` and `Base_SupportBench` datasets, immutable official shared `ExperimentCache`, and a specifically supplied *original* STLAformer checkpoint. It does **not** import TCRR-STLA.

## Main research architecture

- **Pretrained STLA historical encoder:** retains `[B,12,N,D]` temporal latents, multichannel side adapter, and optional STLA partial fine tuning; the official STLA out-step forecast is never used as the v6 prediction input.
- **Audited physical graph:** reconstructs *all plausible direct neighbors* from `core/nodes.csv` and `graph/adjacency.npy`, without Top-4 or inverse-distance fallback. `graph/distance.npy` is audited for valid/invalid/asymmetric entries, but raw numeric values are not assumed to be kilometers. GPS great-circle and validated postmile proxies are encoded separately, with validity flags, and two *candidate* message directions (traffic-forward and disturbance-reverse). Neither direction is claimed to be a verified shockwave direction.
- **TRAIN-only functional graph:** uses unique TRAIN timestamps and one artificial-mask-constrained observed view per timestamp. Uses delayed correlation of *observed traffic changes* after valid-pair filtering, first/second-half sign-stability, per-destination Top-K; functional association is not a causal claim and is **never treated as an extra physical propagation edge**.
- **C1 OCPR:** edge-aware physical and separate functional representation pathways, mask-aware traffic-change features and **lagged, horizon-dependent support reachability on physical edges only**. C1 directly conditions the forecast prior.
- **C2 ECHT:** directional physical graph transports *latent innovations* with forecast-step-dependent edge weights, lag mixture, observation-source support and FP32 sparse accumulation. A separate functional graph provides *statistical prediction context* (not physical transport). Jointly generated future hidden states are directly decoded by a common Speed/Flow head.

## Read-only inputs and separate outputs

Required sibling directories in the repo:

```text
Speed_LLM_RAG/
├── PCFT-STLA/                         # unpacked v6 project
├── Backbone/                          # official STLAformer source + shared-cache API
├── Base_SupportBench/                 # only data interfaces are used
├── Backbone_exps/.../best.pt          # original STLA pretrained checkpoint
├── data/processed/traffident/base_supportbench/
│   ├── speed/TraffiDent_Net_06/       # core, graph, views, split metadata
│   └── flow/TraffiDent_Net_06/
└── ExperimentCache/                   # resolved by the existing Backbone API
```

All v6-derived files are isolated:

- `PCFT-STLA_cache/v6/<dataset>/<target>/...` multivariate *observed-only* training/validation/test inputs;
- `PCFT-STLA_cache/v6/graphs/<dataset>/<target>/...` audited physical/functional graph + source SHA/manifest;
- `PCFT-STLA_exps/<target>/<mode>/<dataset>/v6/...` checkpoints, stage reports;
- `PCFT-STLA_logs/<target>/incomplete/v6/` timestamped PBS logs + GPU telemetry.

Existing `directed_neighbors.npz`, `distance.npy`, Base_SupportBench, Backbone, ExperimentCache, and older PCFT datasets are **never rewritten**.

## Check locally / Gadi BEFORE submitting GPUs

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/PCFT-STLA:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"

python -m pytest -q PCFT-STLA/tests

python -u PCFT-STLA/scripts/preflight_v6.py \
  --target speed --dataset TraffiDent_Net_06 --seed 10 \
  --backbone-checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt

python -u PCFT-STLA/scripts/preflight_v6.py \
  --target flow --dataset TraffiDent_Net_06 --seed 10 \
  --backbone-checkpoint Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt
```

The preflight checks the original official STLA checkpoint, **every frozen cache severity**, physical graph audit, a **64-row TRAIN-only observed-data functional builder sketch**, and all four model stages on a real sample. Success line: `V6_PREFLIGHT_PASS`. It does not prebuild the entire 50,000-row multi-cache; that is built only once by the actual pipeline. If you want to eliminate *every* remaining full-cache preparation risk before queueing, run `--preflight-only` using the pipeline command after assigning appropriate CPU resources, as it constructs the complete cache/functional graph, then tests the model with real observed samples (CPU):

```bash
python -u PCFT-STLA/scripts/run_v6.py --target speed \
  --backbone-checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt \
  --preflight-only
```

**Be aware:** The full preflight processes 50k train windows plus the validation and four test input caches. Run it under an appropriate CPU allocation, not during an overloaded HPC login session. No final test labels are used to select a graph.

## Gadi full 4-stage training and evaluation: one PBS per target

```bash
qsub -N PCFT6_S \
  -v TARGET=speed,SEED=10,BACKBONE_CHECKPOINT=Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt \
  PCFT-STLA/pbs/formal_v6.pbs

qsub -N PCFT6_F \
  -v TARGET=flow,SEED=10,BACKBONE_CHECKPOINT=Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt \
  PCFT-STLA/pbs/formal_v6.pbs
```

Each target independently runs **preflight → v6 cache build/verification → audited graph build → GPU AMP smoke for all stages → GPU batch-size probe → S0 direct → S1 OCPR → S2 transport → S3 coupling → all selected checkpoints × s20/s40/s60/s90 final test → fixed-checkpoint validation interventions → final CSV/JSON**. A single target job owns its whole pipeline. GPU is V100 32 GiB, 12 CPUs, PBS `mem=32GB`, `walltime=48h`. GPU probe considers 32/64/96/128, rejects candidates above 75% GPU reserved VRAM; it is heuristic, not a performance guarantee. Set `BATCH_SIZE=32` in the `-v` values to disable probing.

**Each visible Python console line, traceback line and PBS live-log line is timestamped.** `nvidia-smi` writes its own timestamped CSV measurements. Stage results also contain a `timestamp` in `epoch_history.jsonl`.

### Resume

Within a target's v6 run directory, `last.pt` atomically stores model, optimizer/scaler, RNG, stage and the SHA-locked data/graph identities. The next PBS job may pass `RESUME_CHECKPOINT=/absolute/path/to/last.pt` and use the same `BACKBONE_CHECKPOINT`. The resumed pipeline rejects altered graph, normalization, train manifests or stage configuration. Avoid changing the graph files mid-job. If `V6_TIME_GUARD` triggers, resume using its saved `last.pt`.

## Interpretation limits

- Four stages form a **single progressive training path**. Their test differences are *stage improvements*, not independently retrained ablations. Fixed-checkpoint `disable_c1`, `no_transport`, `remove_functional`, `reverse_direction`, and Lag interventions are cheap diagnostics only.
- TRAIN Functional Graph uses deliberately observed-only **s40** views, not full hidden training observations and never validation/test targets. Multiple overlapped TRAIN views are **not treated as independent** when estimating correlation. It does not claim coverage of the separately evaluated s20/s60/s90 severity distributions through actual additional training views.
- The new graph may improve forecasts but could also degrade them. The corpus here does not contain a completed v6 V100 experiment; there are **no validated Speed/Flow MAE gains** yet.
- v6 implementation follows the discussed TCRR v6.1 graph redesign ideas, but its exact source ZIP was unavailable in this session. The v6 graph builder is a fresh, independently tested implementation and is not claimed to be a byte-for-byte port of TCRR.
