# FARE v5.0.0 Technical Design and Implementation Contract

## Why the v4 selector is replaced

FARE v4.0.4 Test-sample diagnosis on Net-06 gave Speed Full/Random/Shuffled
MAE approximately 1.7169/1.7180/1.7175; Flow Full/Random/Shuffled
19.9450/19.9123/19.9483. These are fixed-checkpoint intervention diagnostics,
not causal estimates of independently retrained retrieval value. Two critical
v4 design flaws: TopK selection lacked direct counterfactual candidate-level
forecast gain training, and evidence tokens replaced the 12-step residual
trajectory with temporal mean features.

v5 is an intentionally incompatible architectural rewrite. Main components:

| Source | Contract |
|---|---|
| `fare/memory.py` | Schema 5 historical OOF paired forecasts and reliability masks; train only |
| `scripts/build_oos_predictions.py` | Produce masked-history and full-history predictions from the same chronologically OOF fold model |
| `fare/retrieval.py` | Multi-route causal coarse recall; retrieve completed episode IDs |
| `fare/transfer.py` | Directional sparse path and lag alignment of residual + evolution; third channel is OOF sensing-failure effect |
| `fare/forecast_utility.py` | Ordered TemporalEvidenceEncoder, gain/sigma/harm ranking, conditional set and NULL, training losses |
| `fare/utility_teacher.py` | Multiple TRAIN-only single-candidate counterfactual forecasts, candidate gain and risk action labels |
| `fare/model.py` | Unified full STLAformer suffix injection, horizon-level gate, deficit predictor |
| `fare/training.py` | Evidence/retrieval/joint/risk/calibration stage training and train-only Teacher replay |
| `fare/analysis.py` | Proper candidate rescore and additional fixed-checkpoint comparison modes |

## Information contract

At inference, a Query may access **only** observed historical speed/flow,
observability and time metadata, topology, its base forecast, and fully
completed past episodes whose future horizons end before the query origin
subject to temporal purge. The Query target is available exclusively in the
training loss, counterfactual Teacher, and downstream offline evaluation.

## Historical Failure Influence

For a completed Train episode `i`, predict with the same chronologically OOF
producer using incomplete and complete historical inputs:

`r_i = y_i - forecast_i(incomplete_history)`

`delta_failure_i = forecast_i(complete_history) - forecast_i(incomplete_history)`

These are separate channels; they are not interchangeable. The metadata must
identify the OOF producer and prove its training cutoff. The complete-history
branch uses the cache's actual available full history, never fabricated labels
for physically missing observations.

## Utility target

For a Train Query `q`, freeze a model snapshot and evaluate candidate singleton
`e` under a fixed evidence integrator:

`U(q,e,h) = abs(y_q(h)-base_q(h)) - abs(y_q(h)-pred_q_with_e(h))`

Positive means the candidate improves forecasting at the horizon. Candidate
labels use real train targets, not similarity, not a weak pseudo-oracle and not
Test data. The candidate set includes state-near episodes, low-scoring episodes,
and randomly sampled legal shortlist episodes. The utility teacher supervises
both harmful and helpful historical episodes.

## Learning objectives

- Forecast loss: valid normalized MAE with direct-mask and tail auxiliaries.
- Gain head: robust Huber regression of counterfactual gains.
- Uncertainty head: Gaussian-inspired heteroscedastic residual penalty.
- Harm head: BCE for measured negative gains beyond margin.
- Pairwise and Listwise rankings: positive vs negative gains among evaluated
  candidates for the **same query and horizon**.
- Deficit head: target is whether ANY evaluated candidate can meaningfully
  help for that query/horizon. This is a conservative *observed* candidate pool
  supervision; it is not a global Memory oracle.
- Risk head: TRAIN-only grid search along frozen candidate-minus-base direction
  supervises safe gate calibration.

## Candidate identity preservation

- The encoder inputs three **ordered 12-step channels** plus per-channel valid
  masks, with convolution, time position, attention pooling, early/late
  distinguishing features and query context.
- Each selected candidate receives its **own token**, not a pooled scalar.
- Its retrieval mass directly scales its attention value. A hard-invalid or
  zero-mass candidate cannot inject a token.
- A softmax over valid candidates **and NULL** controls horizon-wise evidence
  use. A final bounded alpha gate multiplies *non-null mass*. The null-only
  result equals the same-parameter STLA base forecast exactly.

## Evaluation needed for TRC-level claims

The present CPU tests establish functionality only. Paper-level evidence must
show: (a) Full vs No Evidence, (b) learned ranking vs Matched Random, (c) learned
ranking vs Shuffled, (d) identity-preserving temporal encoding vs time means,
(e) OOF sensing-failure channel ablation, (f) NULL/Risk and hard-negatives
ablation, and (g) separate train-budget-matched controlled re-training.

Report paired MAE and RMSE for speed/flow and all s20/s40/s60/s90 and five
failure families, plus valid-label count, Harm Rate and evidence usage. For
ranking, report candidate-positive Recall@M/NDCG@K/Pairwise ranking accuracy
on held-out Validation. When intervals or differences are small, say so.

## Deliberate scope boundaries

The current code does not *prove* its retrieval learned a better selector
without these complete experiments. It does not train the independent
chronological OOF STLA producer checkpoints, and does not have a tested real
Net-06 V100 timing/resource measurement. The checkpoint manifest provenance
must be independently audited. Additional cost-efficient parallel replay or
learned propagation can be researched after the ranking objective is validated.
