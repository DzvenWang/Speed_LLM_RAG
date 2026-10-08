# FARE v1.0.2 — combined speed/flow smoke

Adds `scripts/smoke_all.py` as one entry point for the two independent target smoke runs.

- Both targets are checked sequentially under one command (avoids GPU contention).
- Each target has separate preflight, train, five phase checkpoints, and validation reports.
- Failure of one target does **not** suppress testing the other.
- Exits with nonzero status unless all requested targets pass.
- Emits machine-readable `FARE_SMOKE_SUMMARY` and `FARE_SMOKE_ALL_PASS/FAIL` markers.
- Writes `<project>/FARE_logs/smoke/<dataset>/<run>/smoke_summary.json` and per-target logs.
- Works with Net-01/02/03/04/06 without hardcoded node count or paths.

Run from `Speed_LLM_RAG`:

```powershell
python -u FARE\scripts\smoke_all.py --dataset TraffiDent_Net_06 --seed 10
```

This runs a training-phase smoke, not a full OOF controller or formal s20/s40/s60/s90 test.
