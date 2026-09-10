# Migration to v1.2.1 on Gadi

v1.2.1 fixes a scientific-definition bug in the Train-only predictive dependency graph. In v1.2.0, the series was first subsampled by `sample_stride=12` and then lagged, so lag 1/2/3 unintentionally represented 60/120/180 minutes. v1.2.1 keeps sparse correlation anchors but applies lag on the original 5-min timeline, so lag 1/2/3 represents 5/10/15 minutes.

Because predictive dependency affects predictive masks, support-dependent targets, propagation retrieval and LLM evidence, **rebuild the canonical benchmark from scratch**. Do not reuse v1.2.0 `supportbench` output or retrieval caches.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
STAMP=$(date +%Y%m%d_%H%M%S)

mv TraffiDent_SupportBench TraffiDent_SupportBench_v120_backup_${STAMP}

if [ -d data/processed/traffident/supportbench ]; then
  mv data/processed/traffident/supportbench data/processed/traffident/supportbench_v120_backup_${STAMP}
fi
if [ -d data/diagnostics/traffident/supportbench ]; then
  mv data/diagnostics/traffident/supportbench data/diagnostics/traffident/supportbench_v120_backup_${STAMP}
fi
if [ -d artifacts/supportbench ]; then
  mv artifacts/supportbench artifacts/supportbench_v120_backup_${STAMP}
fi

tar -xzf TraffiDent_SupportBench_v1.2.1.tar.gz
mv TraffiDent_SupportBench_v1.2.1 TraffiDent_SupportBench
cd TraffiDent_SupportBench
chmod +x build_all.sh pbs/*.sh pbs/*.pbs tests/*.py
conda activate TFP
python preflight.py
python tests/run_dependency_lag_smoke.py
qsub pbs/build_supportbench.pbs
```

A correct build log should report predictive dependency lags in minutes, e.g.:

```text
anchors_every=12 steps (60 min); lags=[0, 5, 10, 15] min
```

The Stage 06 audit should include:

```text
predictive_dependency_raw_lag_semantics = PASS
```
