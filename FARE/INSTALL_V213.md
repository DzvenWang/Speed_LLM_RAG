# FARE v2.1.2 -> v2.1.3

This is the **full native FARE project ZIP**, not a code snippet or patch based on v2.0.4. It preserves train scripts, training stages, backbone integration, region hierarchy and Checkpoint compatibility. Original v2.1.1 and v2.1.2 checkpoints can be reused.

1. Back up `FARE/`; do not overwrite `FARE_exps`, `FARE_logs`, `Backbone`, or `Base_SupportBench`.
2. Extract this ZIP into `Speed_LLM_RAG/` so `FARE/` lands at the original path.
3. `python -m pytest -q FARE/tests/test_weight_preserving_audit_v213.py FARE/tests/test_candidate_oracle_v212.py`.
4. Use the **new** `FARE/pbs/oracle_audit.pbs`; it defaults to all weight-preserving and original singleton interventions.
5. First submit Speed Smoke and confirm `FARE213_AUDIT_PBS_PASS`. Then submit Speed/Flow pilots.
6. Inspect `counterfactual_summary.md` and `oracle_summary.md`, not just PBS resource logs.

Read `FARE/README_ORACLE_V213.md` for exact commands, output fields and the retrospective-label caveat. No new learnable module has been added.
