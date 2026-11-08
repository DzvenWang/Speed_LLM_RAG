# TCRR v5.1.0 — Architecture and Implementation Contract

## 1. Research structure and names

1. **C1 / CFRM — Counterfactual Failure Response Modeling**: learn how a query's sensing-failure pattern changes forecasts computed on historical examples.
2. **C2 / SCRT — Support-conditioned Residual Transfer**: identify compatible historical residuals by observation overlap, intervention exposure and expected forecasting utility; combine with a None option.
3. **C3 / CRSI — Counterfactual Response-guided Selective Intervention**: decide which of the Normal and CF transferred corrections should be applied, to which nodes/forecast steps, with separate correction strengths.

The word *intervention* refers to **selective intervention on predictions**, not physical traffic control or causal identification of real-world traffic interventions.

## 2. Data and tensor contracts

`B` batch, `K` number of retrieved candidates (usually 12), `H` history (usually 12), `P` forecast steps (usually 12), `N` node count (Net-06 usually 388); do **not** hardcode node count in architecture.

| Tensor | Shape | Meaning |
|---|---|---|
| `base_prediction` | `[B,P,N]` | frozen STLAformer forecast |
| `context` | `[B,N,D]` | projected STLA latent |
| `normal_residual` | `[B,K,P,N]` | stored historical normal residuals |
| `residual - normal_residual` | `[B,K,P,N]` | query-conditioned historical CF response |
| `candidate_source_valid` | `[B,K,H,N]` | validity under historical original observations |
| `query_mask` | `[B,H,N]` | current observed input positions |
| `candidate_weights` | `[B,K+1,P,N]` | SCRT weighted candidates plus None |
| `proposed_normal`, `proposed_cf` | `[B,P,N]` each | nonlinear attribution of full proposal |
| `normal_gate`, `cf_gate` | `[B,P,N]` each | CRSI correction strengths |

Futures are used only to construct targets in training/validation and diagnostic metrics, never as inputs to the forecasting model during inference.

## 3. C1: CFRM

`v5/counterfactual.py` uses independent Normal and CF branches.

`D_cf = R_failure - R_normal`. The TemporalResponseEncoder applies temporal convolutions on `[normal, D_cf]` per candidate and node. The increment is anchored to the actual CF response: `D_cf=0 => increment=0`. A learned transfer gate, predicted incremental utility and positive-gain probability modulate its amplitude. Correct-vs-shuffled pairs receive a ranking penalty only when ground-truth training utility prefers the correct pairing.

`C1 OFF` sets the CF increment to zero without changing the Normal component. It does not remove missing-input awareness from the frozen Backbone; the experiment isolates the explicit CFRM branch.

## 4. C2: SCRT and exact Normal/CF allocation

`v5/support.py` calculates query/source overlap, exposure to the query's missing pattern, recent overlap, masked-only traffic agreement and functional-neighborhood support, including a no-overlap indicator.

`v5/transport.py` performs a first utility-conditioned ranking, forms a mixture, predicts mixture-aware leave-one-out utility and performs a second ranking. It retains the same final candidate weights for both components:

`m_N = Σ_k w_k · g_k · C_N,k`, `m_CF = Σ_k w_k · g_k · C_CF,k`.

Use the bounded operator `f(z)=c*tanh(z/c)` and total `Δ=f(m_N+m_CF)`. **Do not** use `f(m_N)+f(m_CF)` as the full correction. The symmetric allocation is:

`Δ_N = 0.5 * [f(m_N)+f(m_N+m_CF)-f(m_CF)]`

`Δ_CF = 0.5 * [f(m_CF)+f(m_N+m_CF)-f(m_N)]`

This ensures `Δ_N+Δ_CF=Δ` (subject to float rounding). The decomposition allocates the bounded proposal given final candidate weights; it does not establish a separate causal contribution of C1, because C1 can also change the SCRT weights.

Additional CRSI inputs: normal/CF signed direction consistency, signal magnitude, effective candidate count and cross-component sign conflict. No-overlap or negligible-signal positions get **zero consistency rather than false confidence**.

`C2 OFF` removes the explicit support logits, horizon-specific scale and support utility adjustment. The same retrieval and Normal residual remain available.

## 5. C3: CRSI

`v5/controller.py` produces two learned gates `a_N` and `a_CF` in `[0,1]`:

`Forecast = Base + a_N*Δ_N + a_CF*Δ_CF`.

The C3 structure has three controls:

- **CRC — Counterfactual Response Consistency:** agreement and conflict statistics computed from direction-aware candidate responses, rather than candidate variance alone.
- **SLI — Spatiotemporally Localized Intervention:** coarse Horizon-block features and TRAIN-derived statistical-neighbor pooling, plus a bounded local deviation. This is a functional neighborhood, not automatically upstream/downstream road topology.
- **CCC — Component-aware Correction Control:** separate gates, which can accept Normal and suppress CF, or the reverse.

`c3_variant=scale` uses the original five-scale controller baseline; `component` uses dual gates; `agreement` adds CRC; `localized` adds SLI; `full` combines them. A rotating auxiliary variant forecast loss trains the restricted variants on shared weights. They are useful diagnostic comparisons, but for definitive research contributions independently retrain matched capacity/budget variant models.

`C3 OFF` means `a_N=a_CF=1` and exactly the original SCRT bounded proposal. Legacy fixed Gate=0.75 remains an explicit comparator, not C3 OFF. Validation-calibrated rejection, when active, replaces the full correction by zero and restores the original Backbone forecast.

## 6. Losses and stage schedule

`v5/losses.py` includes: raw-unit forecast MAE+RMSE, SCRT candidate gain/harm, soft utility ranking, CFRM incremental utility and positive-gain classification, correct-vs-wrong CF pair ranking, mixture-aware marginal utility, five-scale reference utility/harm, and CRSI component utility/harm. For two correction components, form the training-only signed targets:

`U_N = |Base-Y|-|Base+Δ_N-Y|`

`U_CF = |Base-Y|-|Base+Δ_CF-Y|`

`U_joint = |Base-Y|-|Base+Δ_N+Δ_CF-Y|`

`I_N_CF = U_joint-U_N-U_CF`

The CRSI head predicts `(U_N,U_CF,I_N_CF)` and the two component harm events. Additional temporal smoothness discourages erratic gate variation without imposing a uniform global gate.

Stage S0 builds a new Mask Adapter; S1 builds train-only retrieval/teacher/replay/base caches; C12 trains C1+C2 with four fixed-gate modes; controller stage freezes C1/C2 and trains C3 with eight-switch exposure plus rotating variant supervision. Optional joint tuning is controlled by the YAML. The stage selector compares validation MAE instead of favoring whichever stage finished last.

Calibration uses `actual_dual_gated_correction` on validation, not a scalar gate multiplied by the original full proposal. The calibration procedure is a temporal holdout heuristic, **not** a distribution-free risk guarantee.

## 7. Three switches and diagnostic outputs

Canonical mode `Mxyz`: `x=C1`, `y=C2`, `z=C3`. All eight modes share one checkpoint, Backbone, query masks and candidate IDs. Four legacy `Mxy` modes use the common fixed Gate for focused C1/C2 tests.

Automatic post-training validation runs C12 four-switch checks, C3 eight-switch checks, Correct/Shuffled/Wrong-mask CF, Coverage-only/Shuffled Support, and the five CRSI variants over an identical deterministic limited validation subset. The C3 variant comparison disables the Full-variant rejection threshold to avoid transferring a separately calibrated decision boundary into competing architectures. Results are saved under the formal run's `evaluation/validation/automated/` directory in separate CSV/JSON files, including `c3_variant_comparison.csv`.

Metrics include Overall/Masked MAE and RMSE, H3/H6/H12 behavior, normal/CF harm and utility, component interaction, directional consistency, correction strengths, correction coverage and tail performance as supported by the evaluation utilities. For paper-quality comparisons, expand to multiple datasets, severities and seeds. Never choose the final model or tune component scales on Test futures.

## 8. Implementation and release checks

- `tcrr_stla/v5/model.py`: three orthogonal switches; C3 variant selection.
- `tcrr_stla/v5/counterfactual.py`: CFRM.
- `tcrr_stla/v5/support.py` and `transport.py`: SCRT plus exact two-component allocation.
- `tcrr_stla/v5/controller.py`: CRSI; `scale_reference.py` retains comparator.
- `tcrr_stla/v5/losses.py`: source-specific utility and harm supervision.
- `tcrr_stla/v5/trainer.py`: stage optimization and dual-correction calibration.
- `scripts/evaluate_v5.py` and `auto_diagnose_v5.py`: switched and matched diagnostic runs.
- `tests/test_v51_crsi.py`: exact allocation, C3 OFF, source-specific gates, response consistency, localization, gradient and N=388 checks.

**Not yet validated:** live Gadi end-to-end training, model improvement, real-world geographical region grouping, formal statistical significance, or strong causal/coverage guarantees.
