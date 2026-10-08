# FARE v1.0.4 — Smoke Retrieval Coverage Hotfix

## Why

In v1.0.3, smoke mode selected the first eight TRAIN sensing views and the
first 16 de-duplicated TRAIN memory windows. The 24-step overlap exclusion
could remove every training candidate, causing TRAIN retrieval_valid_groups=0,
candidate_delta=0 and final_gate=0, even while validation retrieval worked.

## Changes

- Smoke memory now includes 64 unique TRAIN windows sampled evenly across the
  TRAIN period (`memory.sampling=spread`). There is **no** use of val/test in
  constructing memory, and overlap exclusion is unchanged.
- Formal training keeps `memory.sampling=head`, up to 50,000 TRAIN windows.
- Training and validation epoch logs now include `retrieval_valid_groups`,
  `candidate_delta`, and `final_gate_mean`.
- Smoke training fails explicitly with `FARE_SMOKE_NO_TRAIN_RETRIEVAL` if no
  legitimate training retrieval candidates are exercised.
- Per-stage summary includes Base/Candidate/Final MAE, candidate gain,
  retrieval validity and mean gate, not only Final metrics.

## Commands

```powershell
python -u FARE\scripts\smoke_all.py --dataset TraffiDent_Net_06 --seed 10
```

Summarize any v1.0.3/v1.0.4 run without reloading the model:

```powershell
python -u FARE\scripts\report_stages.py --run-dir "FARE_exps\speed\TraffiDent_Net_06\formal_smoke\<run_id>"
```

The old v1.0.3 smoke metrics are **not** a basis for estimating retrieval
improvement. Do not reuse its smoke checkpoints for formal training.