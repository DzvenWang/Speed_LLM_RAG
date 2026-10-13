# Upgrade FARE v2.1.1 -> v2.1.2

This archive contains your **actual uploaded FARE v2.1.1** project, with added native Oracle Audit support. It is *not* the earlier v2.0.4 reference overlay.

1. Keep `FARE_exps/`, `FARE_logs/`, `Backbone_exps/`, `Base_SupportBench/` intact; back up original `FARE/`.
2. Extract archive into `Speed_LLM_RAG/` so it contains `FARE/`.
3. Run `python -m pytest -q FARE/tests/test_candidate_oracle_v212.py` and `python -u FARE/scripts/check_metric_protocol.py`.
4. Use `FARE/pbs/oracle_audit.pbs` with a frozen v2.1.1 `candidate_best.pt` checkpoint.
5. Smoke first, then Speed/Flow 12-query Top64 jobs. Read `FARE/README_ORACLE_V212.md` for full commands and interpretation.

No past checkpoint is overwritten and no model retraining is needed.
