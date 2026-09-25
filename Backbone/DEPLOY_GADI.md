# Gadi deployment

The Gadi PBS entry point has a safe default root and does **not** require
`SPEED_LLM_RAG_ROOT` to be exported before `qsub`.

Default locations:

- project root: `/scratch/mh20/zw1165/Speed_LLM_RAG`
- TFP environment: `/scratch/mh20/zw1165/envs/TFP`
- live logs: `/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone_logs`

The normal formal seed-10 job can therefore be submitted directly:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
mkdir -p /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone_logs
qsub pbs/run_patchstg.pbs
```

The PBS defaults are `STAGE=formal` and `SEED=10`. Override them when needed:

```bash
qsub -v STAGE=formal,SEED=20 pbs/run_patchstg.pbs
qsub -v STAGE=formal,SEED=30 pbs/run_patchstg.pbs
qsub -v STAGE=development,SEED=10 pbs/run_patchstg.pbs
```

Each job creates a live log like:

```text
Backbone_logs/PatchSTG.formal.seed10.<PBS_JOBID>.<timestamp>.live.log
```

Every complete stdout/stderr line in that live log is timestamped. The PBS job
uses `/scratch/mh20/zw1165/envs/TFP/bin/python` directly instead of relying on
`conda activate` inside a non-interactive batch shell.

For local/portable execution, the project root is still inferred automatically,
or can be overridden explicitly:

```bash
export SPEED_LLM_RAG_ROOT=/path/to/Speed_LLM_RAG
python scripts/run_experiment.py --stage formal --seed 10
```
