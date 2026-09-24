# v0.9 Implementation Notes

## Research scope

v0.9 deliberately does not train a new forecaster, LLM, or forecast-utility classifier. It is a diagnostic used before implementing the formal PreSuRe fragment composer.

## Why fragment-level analysis

The existing whole-patch completion path can force one historical episode to explain all missing values in a damaged patch. v0.9 tests whether the useful information is actually distributed across multiple historical episodes and smaller local regions.

## Safety against leakage

- Candidate pool: train split only.
- `global_top1`, `local_observed`, and `local_softmax`: no hidden query state, no future target.
- `oracle_completion`: hidden history truth only; diagnostic-only.
- No v0.9 selector uses future `Y`.

## Computational profile

The expensive retrieval stage is reused from the existing run. v0.9 constructs fragment compositions on CPU and evaluates only the resulting composed inputs through frozen PatchSTG. It does not forward every candidate through PatchSTG, making it much cheaper than v0.6/v0.8 candidate-level forecast-utility diagnostics.
