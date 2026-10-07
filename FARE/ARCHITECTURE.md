# FARE v0.2.3 — Strong Restoration + Explicit Forecast-Utility Control

FARE remains a four-expert diagnostic model with no MoE router. The expert architectures remain Context, Structural TMCP, Forecast-Utility Retrieval and Backbone-Aware. v0.2.3 changes the training mechanism so that restoration capability and forecast intervention are learned separately.

For each neural expert `e`:

1. **Stage A:** learn a dense raw estimate `X_hat_e` while the prediction-control head is frozen.
2. Build formal train/val counterfactual utility curves by evaluating frozen STLAformer at `alpha in {0,.25,.5,.75,1}`:
   `X(alpha)=X_obs + M_artificial * alpha * (X_hat_e-X_obs)`.
3. Label `alpha* = argmax_alpha Utility(alpha)` where utility is baseline-relative downstream forecast MAE improvement.
4. **Stage B:** freeze the restorer and train the cell-wise gate. Its query mean is explicitly supervised toward `alpha*`, while forecast/no-harm losses permit cell-level deviations.

This removes the `g=0` local optimum that appeared in v0.2.2 without returning to sparse Top-32 restoration. Every missing cell still receives a dense estimate.

## Retrieval ranking

The Retrieval candidate labels remain actual downstream STLAformer utility labels from formal train/val. Ranking is now pairwise:

`U_i > U_j  =>  score_i > score_j`

using logistic pair loss. This is invariant to small absolute utility magnitudes. Random-order reference: loss `ln(2)=0.6931`, accuracy `0.5`.

Retrieval Stage A jointly learns target/node candidate scoring, support fusion and raw restoration. Retrieval Stage B freezes them and trains only the final prediction-control gate from alpha-utility labels.

## Structural expert

The Structural expert remains causal 28-day same-slot TMCP. Its raw dense restoration is non-parametric, so intervention alpha continues to be selected once on formal validation using downstream forecast MAE.

## Evaluation

Report raw restoration MAE, controlled forecast MAE/RMSE/gain, Expert Oracle, Safe Oracle, per-family routing, control gate statistics, counterfactual alpha diagnostics and Retrieval pairwise ranking diagnostics.
