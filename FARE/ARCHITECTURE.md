# FARE v0.2.4 Architecture

For expert `e`, Stage A produces dense raw restoration `X_hat_e` for all artificial-missing cells.

For each training/validation query, the frozen STLAformer evaluates the counterfactual intervention grid:

`U_q(alpha) = (MAE_base - MAE_q(alpha)) / MAE_base`

for `alpha in {0, .25, .50, .75, 1}`.

The Stage-B query controller predicts the entire scaled utility vector rather than regressing only `argmax alpha`. Its losses are:

- utility-curve Smooth-L1 regression,
- best-alpha cross entropy,
- pairwise alpha-order ranking.

A separate cell reliability head is supervised from raw restoration error on artificial-missing cells. Let `r_qtn` be the learned reliability and `a_q` the alpha selected by the query landscape. The controller normalizes reliability on missing cells and forms:

`g_qtn = clip(a_q * normalize(r_qtn), 0, 1)`

`X_controlled = X_obs + g * (X_hat - X_obs)` on artificial-missing cells only.

During Stage-B training, forecast/no-harm gradients use oracle best-alpha as an override so the reliability head can learn useful allocation without depending on an immature query selector. At validation/inference the override is absent and the model uses its own predicted landscape.

Observed and originally-invalid cells are immutable.
