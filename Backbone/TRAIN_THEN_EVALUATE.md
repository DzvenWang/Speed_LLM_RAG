# PatchSTG formal train then evaluate

Default GADI formal submission:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
qsub -v STAGE=formal,SEED=10 pbs/run_patchstg.pbs
```

The job performs, in order:

1. Core-5 s40 formal training.
2. Verify the exact run `best.pt`.
3. Evaluate the checkpoint on Core-5 s20, s40, s60 and s90 caches.
4. Print `FINAL SEVERITY SUMMARY` and `FINAL SCENARIO SUMMARY` in the same PBS live log.

Formal auto-evaluation is enabled by default. To train only:

```bash
qsub -v STAGE=formal,SEED=10,AUTO_EVAL=0 pbs/run_patchstg.pbs
```

Development remains train/validation only:

```bash
qsub -v STAGE=development,SEED=10 pbs/run_patchstg.pbs
```

If training fails, evaluation is not started. If any severity evaluation fails, the PBS job exits non-zero.
