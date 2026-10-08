# v0.2.4 design decision

v0.2.3 showed three neural controllers converging to their dataset mean alpha rather than learning conditional control. This version changes the supervision geometry rather than tuning loss weights.

## Query level

Predict the complete counterfactual downstream forecast-utility curve over the intervention grid. The query head is supervised by curve regression, best-alpha classification, and pairwise ordering. Alpha=0 remains in the action set, so abstention is an explicit learned action.

## Cell level

Predict restoration reliability only. Reliability is normalized over artificial-missing cells, so it redistributes the query-level alpha without redefining the amount of intervention. Local supervision comes from train/validation restoration correctness; no evaluation truth is used.

## Why this is different from a dense gate

A scalar query label no longer supervises thousands of cell gates. The query decision and local allocation are separate estimands with separate labels. This directly addresses the mean-gate collapse observed in v0.2.3.
