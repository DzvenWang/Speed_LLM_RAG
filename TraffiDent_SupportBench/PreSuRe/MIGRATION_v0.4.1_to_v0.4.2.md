# PreSuRe v0.4.1 -> v0.4.2

This revision changes PBS scheduling only. Model/data/scientific logic is unchanged.

## Single-GPU full pipeline

`pbs/full_pipeline.pbs` now runs the complete PreSuRe pipeline sequentially in a single PBS allocation and on a single V100:

1. Preflight
2. PatchSTG backbone artifact export
3. Retrieval + sensing-conditioned node-feature construction
4. Oracle A -> B -> C -> E -> F -> D
5. M0 -> M1 -> M2 -> M3 -> M4 -> M5
6. Summary generation

Both `scripts/submit_development.sh` and `scripts/submit_formal.sh` submit exactly one GPU job.

The older per-phase and pooled-suite PBS files are retained only for manual debugging and continuation. They are not used by the default full-pipeline submission scripts.
