# PreSuRe v0.4.0 -> v0.4.1

This revision changes PBS scheduling only; model/data/scientific logic is unchanged.

- Oracle A/B/C/E/F/D now run sequentially in `pbs/oracle_suite.pbs` using one V100 allocation.
- M0/M1/M2/M3/M4/M5 now run sequentially in `pbs/variant_suite.pbs` using one V100 allocation.
- After shared backbone/retrieval prerequisites, the Oracle suite and Variant suite can run concurrently on two V100s.
- Live logs remain under `PreSuRe_logs/`; PBS `.OU` files remain under `PreSuRe_logs/out/`.
- `pbs/oracle_support.pbs` and `pbs/variant.pbs` remain available for manual single-task debugging.
