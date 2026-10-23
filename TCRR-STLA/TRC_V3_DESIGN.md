# TCRR-STLA v3.0 — TRC research implementation

**Primary aim:** improve Speed/Flow forecast MAE/RMSE under incomplete sensing, *and* make the gain attributable to three identifiable changes. Not an assertion that any unrun experiment already improves on v2.0.

## Three contributions

**C1 Paired Counterfactual Failure-Response Memory.** The fixed Teacher predicts the **same Train history** under complete and query-matched masked observation. Historical `R_normal = Y_train - Teacher(X_train, 1)`, `R_masked = Y_train - Teacher(X_train * M_query, M_query)`, `D_failure = R_masked - R_normal`. `CounterfactualResidualTransfer` receives these values. `M0` uses normal errors; `M1` adds paired failure response. Train-only query supervision also predicts `Teacher(X_query_full,1)-Teacher(X_query_obs,M_query)`. This is an auxiliary teacher-response target, **not** a true-future prediction label; it never appears in inference.

**C2 Support-Conditioned Residual Transfer.** `v3/transfer.py` compares query and candidate history only on *actually observed* coordinates, with support coverage, missing age, neighbor coverage and valid adjacent-step trends. Neighbors come from the existing train-only statistical graph; do not claim causal direction. A learnable candidate re-ranker adjusts the shared similarity shortlist and learns listwise relative utility versus None. `M2` adds this to `M1`.

**C3 Gain–Harm-Calibrated Bounded Correction.** The common evidence-only reference gate is retained as the control for M0–M2. M3 changes only final gating through supervised expected forecasting gain, harm magnitude and uncertainty; corrected forecasts are `base + gate * bound(weighted historical residuals)`. No free-form future head exists in the full retrieval pathway. The test-time forward takes no true query future or full query history.

## Independent contribution matrix

| Variant | C1 paired response | C2 support-conditioned transport | C3 gain–harm correction |
| --- | --- | --- | --- |
| M0 | no; plain historical normal error | no | common reference gate |
| M1 | yes | no; similarity shortlist | common reference gate |
| M2 | yes | yes | common reference gate |
| M3 | yes | yes | gain–harm calibrated controller |

Train all four **independently** using identical frozen STLAformer checkpoint hash, dataset, target, seed, train-only memory, shortlist size, learning rate, epochs and batch size. Parameter modules are instantiated across levels (unused modules do not affect forward). The differences are **conditional** contributions. Add shuffled CF pairing, no support conditioning and no gain–harm inference interventions; interpret these as within-checkpoint sensitivity, never as substitutes for independently trained models.

## Training, losses, information boundaries

- Backbone base frozen by `schedule: frozen`. Learnable mask adapter remains common across M0–M3.
- Frozen Teacher generates matched historical errors; both complete/masked calls use identical Teacher parameters.
- Train `*_x_full_norm.npy` is loaded *only* with `cf_training=True` on Train split. Evaluation `SupportSplit(...,'test')` never opens complete query history. Val early stopping uses true future only for metrics, not a query-complete teacher label.
- Main supervised objective uses original-unit MAE (+ original protocol RMSE); v3 adds optional train-only paired CF recovery, listwise candidate gain/harm supervision, and correction harm-tail supervision.
- `candidate_y` is retrieved **historical TRAIN future**, not the current query's future. Time exclusion remains enforced in the original shortlist cache generation.
- Label-based candidate gain is constructed in `v3/losses.py` *only during Train*; it is not used to choose candidates in inference.
- Corrected forecast depends on retrieved historical residual; zeroing evidence returns the same Base forecast.
- The query/full Teacher target can be unreliable if a mostly-s40 pretrained model is given all-ones masks out of distribution; track CF Recovery and actual MAE independently.

## Main code

`model.py` routes only v3 checkpoints to `v3/transfer.py`; `trainer.py` adds labels and losses only on train batches; `forecast_pool.py` fixes **per-query coverage** and makes configured traffic score weight functional; `scripts/train.py --contribution M0..M3` saves isolated directories; `scripts/summarize_v3_attribution.py` checks matching protocols and extracts incremental MAE changes.

Inherited v2 code is retained for compatibility/auditing, not all v2 traffic utility heads participate in v3 forward.

## Explicit inference mechanism tests

- C1: `--evidence-mode no_mask_replay`, `no_failure_residual`, `shuffled_cf_pairing`.
- C2: `--evidence-mode no_support_conditioning` versus `full`; preserve exact same selected candidates for a fair reranking test.
- C3: `--evidence-mode no_gain_harm` or `no_risk`; these are **inference interventions**.
- Negative control: `shuffled` and `cross_history` perturb retrieved evidence. A trained No-Retrieval baseline requires a second training run, not turning correction off at test.

## Limitations still requiring Gadi verification

Unit tests cover finite synthetic forward/backward and code-path isolation but **do not establish real Net-06 MAE gains, V100 peak memory, or 48h completion**. Full training/validation/evaluation must be run on actual Base_SupportBench / Backbone and official STLAformer vendor weights. v3 checkpoint is deliberately incompatible with loading v2 optimizer states.
