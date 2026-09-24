# PreSuRe-P1 Design

P1 is a **go/no-go retrieval-value pilot**, not the final PreSuRe architecture.

## Question

Does causally available historical episode evidence improve a frozen PatchSTG beyond the gain obtainable from adding the same small residual adapter alone?

## Architecture

```text
Incomplete query -> Frozen PatchSTG -> base forecast + pre-head state
                                      |
                                      +-> PCA key -> train-only causal Top-K memory
Historical train episodes ----------------------^       |
                                                        v
                               aggregated history/evolution evidence
                                                        |
                                   zero-init node-wise residual adapter
                                                        |
                                      final = PatchSTG + residual
```

PatchSTG is never modified and no retrieved value is written into its input.

## Four trainable controls

All controls use the **same adapter architecture and parameter count**:

- `adapter_only`: support channels are zero.
- `shuffled_support`: causality-preserving random historical episodes.
- `history_only`: real retrieved support, using only historical-state summaries.
- `history_future`: real support plus its already-observed subsequent trajectory and future coverage.

The frozen PatchSTG itself is the fifth baseline and has no P1 training.

## Retrieval contract

P1 intentionally keeps retrieval simple. The key is a low-dimensional PCA projection of the frozen PatchSTG pre-head node state, globally pooled and cosine-normalized. P2, not P1, will make retrieval explicitly mask-aware.

For every candidate episode j used by query q:

`candidate_start + history + horizon < query_start`

Thus the candidate's subsequent trajectory has already occurred. Historical future values are used only where `future_valid` is true; the corresponding coverage is supplied to the adapter.

## Why no Gate / Need / Utility Teacher

P1 first tests whether retrieval evidence has independent value. A gate could hide a failed retrieval branch by collapsing to PatchSTG, and a utility teacher would add a circular dependency before fusion has been validated.
