# FARE v2.0.0 — Forecast-Oriented Retrieval Integration

This is a full source project based on FARE v1.3.4. It retains the original five-stage trainer, STLAformer integration, OOF calibration, PBS scripts, stage evaluation, P0.1 diagnostics, and preflight infrastructure. This release has **not** been trained on the real datasets; no MAE/RMSE improvement is claimed.

## Implemented v2 pathways

1. **Support Demand Query:** `fare/support_demand.py` computes an inference-safe, learned region-level forecast/mask/hidden-state embedding, fused into coarse retrieval queries. It does not consume target labels.
2. **Historical Future Memory:** `TrainMemory` loads, saves, and validates target-normalized TRAIN future trajectories and their validity masks, separately from the observed historical inputs.
3. **Causal Retrieval:** when historical future is present, candidates can only be selected if their full recorded input+future window ended before the query was issued. The original no-overlap exclusion remains.
4. **Forecast-Conditioned Utility Features:** the retrieval re-ranker receives differences between historical-future region profiles and the Backbone's current *predicted* region future. It never sees current ground-truth future values.
5. **Forecast Utility Ranking Loss:** training-only, candidate-wise historical analog forecast errors versus current training labels supply an auxiliary listwise teacher. This is a surrogate utility teacher, **not** a full counterfactual STLAformer loss.
6. **Analog Forecast Adapter:** the selected retrieved historical future profiles yield a learnable residual proposal in forecast space, which passes through the existing utility/risk-gated final predictor. Its last layer starts at zero for conservative initial behavior.
7. **Full project preservation:** unchanged Gadi PBS train/evaluation entry points and prior P0.1 infrastructure are retained.

## Important boundaries

- Forecast Utility targets use a **persistence-adjusted analog surrogate** and are not full-model OOF counterfactual estimates. The previous OOF controller remains present. No node-specific candidate retrieval, separate complete-data teacher, or new six-stage trainer is claimed.
- `stratified_temporal` currently implements deterministic time-uniform coverage, NOT congestion-stratified sampling. The name is retained as a configuration alias and must not be misrepresented in research reports.
- An old v1.3 memory file has no historical future fields. For v2 full mode, rebuild memory with `FARE/scripts/build_memory.py` or allow the training runtime to build it automatically. Do not resume v1.3 model checkpoints using the changed v2 architecture: start a fresh run.
- Actual benchmark integration requires adjacent `Base_SupportBench` and `Backbone` projects, the formal caches, and STLAformer checkpoints. Their artifacts are not contained in this ZIP.

## Run on Gadi

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
python -u FARE/scripts/preflight.py --target speed
python -u FARE/scripts/preflight.py --target flow
python -u FARE/scripts/train.py --target speed --seed 10 --smoke
python -u FARE/scripts/train.py --target flow --seed 10 --smoke
qsub -N FARE_v2_speed -v TARGET=speed,SEED=10 FARE/pbs/train_formal.pbs
qsub -N FARE_v2_flow -v TARGET=flow,SEED=10 FARE/pbs/train_formal.pbs
```

PBS remains `gpuvolta`, 1 V100, 12 CPUs, 32 GB RAM, and 48h, with per-line timestamped live logs and optional in-allocation auto evaluation.

## Development validation

- `pytest -q tests/test_v2_components.py tests/test_smoke_memory.py tests/test_full.py -k 'not staged_training'`: **10 passed**.
- End-to-end full staged evaluation test cannot complete here because `experiment_common.metrics` lives in the external Backbone/Base_SupportBench repository not present in the supplied zip. The actual train/val/test Gadi pipeline is therefore **not verified** in this environment.

## Required TRC ablations

Always measure matched retraining of no-memory, random-memory, similarity-only, historical-future-off, utility-ranking-off, demand-off, and full retrieval. Evaluate MAE/RMSE at s20/s40/s60/s90 for both speed and flow, not merely the auxiliary retrieval loss.
