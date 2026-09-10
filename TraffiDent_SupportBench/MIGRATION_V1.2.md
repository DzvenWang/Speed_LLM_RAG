# Migration to v1.2.0 on Gadi

v1.2.0 changes benchmark semantics (predictive dependency, predictive sensing families, memory stride=6 and three-track protocols). Rebuild canonical data from scratch; do not reuse v1.1.x benchmark outputs.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
STAMP=$(date +%Y%m%d_%H%M%S)

# Keep old code/data for rollback.
mv TraffiDent_SupportBench TraffiDent_SupportBench_v11x_backup_${STAMP}
if [ -d data/processed/traffident/supportbench ]; then
  mv data/processed/traffident/supportbench data/processed/traffident/supportbench_v11x_backup_${STAMP}
fi
if [ -d data/diagnostics/traffident/supportbench ]; then
  mv data/diagnostics/traffident/supportbench data/diagnostics/traffident/supportbench_v11x_backup_${STAMP}
fi
if [ -d artifacts/supportbench ]; then
  mv artifacts/supportbench artifacts/supportbench_v11x_backup_${STAMP}
fi

# Unpack release placed in PROJECT_ROOT.
tar -xzf TraffiDent_SupportBench_v1.2.0.tar.gz
mv TraffiDent_SupportBench_v1.2.0 TraffiDent_SupportBench
cd TraffiDent_SupportBench
chmod +x build_all.sh build/*.py derived/*.py pbs/*.sh pbs/*.pbs tests/*.py

conda activate TFP
export PYTHONPATH=$PWD:$PYTHONPATH
python preflight.py
qsub pbs/build_supportbench.pbs
```

Successful build ends with:

```text
STAGE 06 DONE: benchmark audit
STAGE 07 DONE: canonical API smoke
TraffiDent-SupportBench build completed
PBS job exiting rc=0
```
