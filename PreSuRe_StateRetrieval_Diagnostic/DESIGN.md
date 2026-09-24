# v0.9 Design — Retrieval-Augmented Fragment Completion Diagnostic

## Motivation

PatchSTG is already a strong numerical forecaster. Generic RAG provides information that largely overlaps with patterns already learned by the backbone. The remaining retrieval opportunity is narrower: **recover information lost by incomplete sensing**.

Therefore retrieval is evaluated only as a missing-state completion mechanism.

## Diagnostic decomposition

Let the incomplete history be `X_obs`, observation mask `M`, and hidden complete history `X_full` (available only for controlled diagnostic evaluation).

For a damaged PatchSTG patch `P`, the existing retriever produces a high-recall train-only Top-K pool:

`R(P) = {r_1, ..., r_K}`.

v0.9 asks whether the missing target region should be completed from one whole historical source or from multiple smaller fragments.

### Fragment geometries

- whole patch;
- temporal block;
- node trajectory;
- node x temporal block;
- cell-level upper bound.

For fragment `f`, a candidate is evaluated only on cells in `f`.

### Completion Oracle

`oracle_completion` selects the historical fragment minimizing error against the hidden **history** state. It is allowed to abstain if no candidate improves the original incomplete input. It does not see the future prediction target.

This answers whether useful completion evidence exists in the retrieved pool.

### Observable fragment discovery

`local_observed` scores candidates using only observed context adjacent to fragment `f`:

- observed levels in the same PatchSTG patch;
- local temporal differences;
- the same affine state alignment used by the existing transport path.

This answers whether useful fragments are discoverable before the missing state is revealed.

### Fragment composition

Different fragments may select different historical cases. Composition writes only to missing/restorable cells. `local_softmax` additionally combines multiple candidates inside each fragment rather than hard-selecting one source.

## Primary comparisons

1. `oracle_completion:whole_patch`
2. `oracle_completion:temporal*`
3. `oracle_completion:node_track`
4. `oracle_completion:node_block3`
5. `oracle_completion:cell` (ceiling only)
6. observable versions of structured fragments

Both completion MAE and frozen PatchSTG forecast MAE are reported.

The key diagnostic gap is:

`structured fragment Oracle - whole-patch Oracle`.

A positive gap supports a learned fragment composer in the subsequent PreSuRe model.
