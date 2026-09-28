# Primary seed10 failure-recovery hotfix (2026-09-26)

This patch targets the failures observed in the Core-5 seed10 Formal run batch without changing the shared Protocol, cache identity, model hyperparameters, or Baseline project version (still 1.1.0).

## Fixed

- GinAR: validation divergence after an already-selected finite Val-best checkpoint now stops training and preserves that checkpoint.
- RAST: evaluation unwraps `predict(...) -> (raw_prediction, impact)` before saving predictions.
- SARAF: runner diagnostics no longer overwrite canonical identity metadata; evaluation uses `RUN_MANIFEST.json` as the authoritative identity artifact when available.
- ReMFT: evaluation passes `use_amp` and uses `formal_eval_batch_size`.

## Existing run reuse

After deploying the patch, already-trained RAST, SARAF, and ReMFT Formal runs can be evaluated without retraining:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
bash scripts/submit_evaluation_recovery_seed10.sh
```

GinAR must be rerun because the failed trajectory intentionally did not write `best.pt`/`summary.json` before this patch. Submit only GinAR:

```bash
qsub -v METHOD=GinAR,SEED=10 pbs/methods/formal_primary.pbs
```

Run the smoke test first:

```bash
/scratch/mh20/zw1165/envs/TFP/bin/python tests/smoke_primary_failure_recovery_20260926.py
```
