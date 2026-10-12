# FARE v2.0.3 — Forecast-Utility Retrieval Pilot

Derived from `FARE_v2.0.2_Pilot_Ready.zip`. This is a **new research implementation**, not a proven accuracy improvement. Existing Base_SupportBench, Backbone/STLAformer, memory cache, Gadi PBS, logging and official evaluation interfaces are retained. Target is Speed/Flow on Net-06, later Net-01/02/03/04/06.

## Motivating evidence from v2.0.2 pilots

- On Net-06, BS32 finished all phases, while BS64 ran out of GPU memory during Full.
- Injection-stage validation had the strongest MAE (Speed ~1.5767, Flow ~17.3641); later Partial/Full deteriorated.
- Top-8 retrieval weights remained close to uniform (effective K ~8), and Selected vs Uniform ranking errors were similar.
- Final improvements are **not sufficient evidence of Retrieval gains** without real/random/no-memory comparison.

## Actual code changes

1. **Causality before coarse Top-K** (`fare/retrieval.py`): when using historical future trajectories, excludes candidates whose historical outcomes were not available at query issuance *before* searching the Top-64, not after the pool has already been chosen. The previous behavior could fill the coarse pool with ineligible future candidates and discard useful earlier ones.
2. **Historical-future alignment** (`fare/retrieval.py`): for each candidate, estimate a bounded current-vs-historical level offset using only recent common observed anchors. Apply it to the candidate's already-completed regional future trajectory.
3. **Forecast-utility head trained directly over all Top-64 candidates** (`fare/forecast_utility.py`): region-level normalized MAE improvement relative to the same frozen-base forecast is the TRAIN-only supervisory proxy. Optimize signed utility regression, positive/negative classification, best-vs-worst pairwise discrimination, and soft ranking regret. Head predictions, not future labels, influence test-time reranking. Keeps existing selected Top-8 ranking and state reconstruction objectives.
4. **Runtime retrieval alternatives** (`scripts/evaluate.py`, `scripts/ablate.py`): `similarity_memory` and `utility_off` added to the existing `full`, `random_memory` and `no_memory` modes. Same checkpoint + validation dataset comparison.
5. **Automatic Pilot retrieval attribution**: pilot PBS invokes post-train Val ablations and writes `retrieval_attribution/ablation_summary.csv` and `retrieval_attribution/retrieval_gain.md`. Positive results are not proof of causal gain; matched retraining and Test multi-seed are still required.
6. **Conservative stage control**: `pilot.yaml` trains Retrieval 8 / Recovery 5 / Injection 12, **stops before Partial/Full**. `pilot_finetune.yaml` allows all five phases, freezes Retrieval during fine-tuning, drops learning rates and reloads the strongest earlier checkpoint before each fine-tuning stage.
7. **Intermediate utility diagnostics**: coarse positive eligibility, utility sign prediction accuracy, oracle positive group fraction, oracle vs weighted expected gain, pairwise loss, regret, alongside resource telemetry and existing forecast diagnostics.

## No future leakage

- Memory comes from *training windows only*; memory historical future must have ended by the current query issue time.
- No `y_raw`, `y_norm`, `x_full`, `future_valid` from the current query enters model inference forward or retrieval candidate scoring.
- The actual forecast target is accessed only inside TRAIN loss/Validation diagnostic and checkpoint evaluation; it never determines retrieval at inference.
- Historical-future memory is not current-query ground truth. The utility target is an auxiliary **regional proxy** (mean future outcomes vs mean Backbone forecast), not the final nodal correction oracle.

## Limitations and protocol

- v2.0.3 is **not backward-compatible for training resume** from a v2.0.2 run; the objective and inference behavior changed. Start a new Pilot and keep old logs/checkpoints for comparisons.
- The original STLAformer architecture and upstream SupportBench input schema are unchanged. No external plugin or other experimental project's source code is required.
- BS32 recommended for Gadi V100; BS64 OOM in the v2.0.2 pilots.
- `full.yaml` is provided for later formal studies, but **do not submit Formal until Pilot validates retrieval gain**.
- Changes may improve, not necessarily guarantee, MAE/RMSE. A genuine retrieval gain requires Full < Random, Similarity-only, Utility-off, and No-memory on held-out Val with matched protocol. Retrained ablations and test split remain essential.

## Tests

`pytest -q FARE/tests` from repository root. Current container does not include `experiment_common.metrics` from the external Base_SupportBench; tests requiring this import were run against a temporary **test-only** metrics stub, not shipped. All new utility/causality tests can be run without it.
