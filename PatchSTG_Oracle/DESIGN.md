# Design contract

## Primary contrast

`PatchSTG(X_incomplete)` versus `PatchSTG(X_oracle_complete)` using one immutable checkpoint.

`X_oracle_complete` restores only benchmark-hidden history cells for which the canonical raw-valid mask confirms a real observation exists. No future value is used as model input.

## Source audit

Before GPU evaluation the project verifies:

1. canonical `traffic_2024.npy` speed channel and raw-valid mask have the expected node count;
2. `window_start` produces a history window whose normalized values match cache `x_obs_norm` on observed cells;
3. observed cache cells are raw-valid;
4. recoverable missing cells exist;
5. replacing all raw-valid cells is numerically equivalent to replacing only hidden raw-valid cells, proving that the cache's observed values and canonical values agree.

## Deep representation diagnostic

The official vendor tree is never modified. A non-invasive forward reproduces PatchSTG and exposes:

- L3 restored node representation
- L5 / pre-head representation

The first evaluation batch must satisfy official-forward/deep-forward and pre-head reconstruction equivalence.

For each query the test stores relative L2 and cosine gaps between incomplete and complete L3/L5 states, then correlates those gaps with the actual oracle forecast improvement.

## v1.1 repair experiments

The complete-history oracle showed that missing current state contains useful information, especially at short horizons, while earlier historical-future retrieval provided little incremental information to a strong PatchSTG. Version 1.1 therefore tests the input semantics and state representation before adding another retrieval system.

### Shared observation semantics

For each history cell, the adapter receives four causal features:

1. whether the value is observed;
2. the current consecutive missing-run length;
3. the age of the last observation at the forecast origin;
4. the fraction of the node's history that is observed.

A temporal convolution uses PatchSTG's own temporal patch size and produces an additive embedding with the same shape as the official value/time/node embedding. Its final projection starts at zero. PatchSTG's vendor source is unchanged.

### Controlled variants

- `mask_aware`: train only the observation adapter with forecast MAE.
- `state_aux`: train the same adapter and a reconstruction head with `forecast_MAE + 0.2 * missing_state_MAE`.

The backbone and regression head remain frozen by default. The state head reads the shared pre-head representation and has no edge back into the forecast as a value input.

### True coverage intervention

The universe of restorable cells is `artificial_missing AND isfinite(x_full_norm)`. A query-indexed seeded permutation defines one order over this universe. Fractions take prefixes of that order, which guarantees nested restored sets. Selected cells receive their actual `x_full_norm` value and switch from unobserved to observed. Raw-invalid cells cannot be selected.

### Falsifiable interpretation

- Mask-aware gain at 0% restoration tests whether PatchSTG mainly lacks missingness semantics.
- State-aux gain over mask-aware tests whether explicit current-state representation learning adds value.
- Coverage-curve slope tests whether each model can use additional genuine observations.
- A flat learned-model curve together with a strong reference oracle suggests a remaining architectural bottleneck; a strong curve with weak 0% gain suggests information absence rather than missingness ambiguity.
