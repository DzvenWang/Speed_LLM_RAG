# Official Sources — Baseline v0.6.0

Pinned repositories and commits are defined in `configs/official_sources.yaml` and verified by `scripts/source_lock.py`.

v0.6.0 changes the shared experiment infrastructure, not the baseline method architectures. Method-specific adaptations remain explicit in the adapters/runners and must not alter frozen Shared Protocol membership or ExperimentCommon metric semantics.
