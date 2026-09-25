# Gadi PatchSTG PBS fix (v1.0.1)

Root cause of job 179848127: the old PBS script exited before Python because it required `SPEED_LLM_RAG_ROOT` to be exported into the batch environment.

The updated `pbs/run_patchstg.pbs`:

- defaults to `/scratch/mh20/zw1165/Speed_LLM_RAG` on Gadi;
- defaults to `STAGE=formal`, `SEED=10`;
- runs `/scratch/mh20/zw1165/envs/TFP/bin/python` directly;
- writes one live log under `Backbone_logs/`;
- timestamps every complete stdout/stderr line;
- keeps `SPEED_LLM_RAG_ROOT`, `TFP_ENV_ROOT`, `STAGE`, and `SEED` overridable.

Submit the normal formal seed-10 job:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
mkdir -p /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone_logs
qsub pbs/run_patchstg.pbs
```

Other formal seeds:

```bash
qsub -v STAGE=formal,SEED=20 pbs/run_patchstg.pbs
qsub -v STAGE=formal,SEED=30 pbs/run_patchstg.pbs
```

Watch the live log:

```bash
tail -f /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone_logs/PatchSTG.formal.seed10.*.live.log
```
