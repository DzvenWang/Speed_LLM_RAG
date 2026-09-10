# Migration to v1.2.2 on Gadi

v1.2.2 is the final support-oriented benchmark refinement. Canonical time splits, dense windows, general/main-support tracks, OOD definition, 5-min dependency lag semantics and memory stride=6 remain unchanged. The predictive structured layout catalogues are rebuilt because their construction objective changed.

## Rebuild requirement

Do not reuse the v1.2.1 benchmark directory for formal v1.2.2 experiments. Predictive layouts and their resolved view IDs must be regenerated together.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
STAMP=$(date +%Y%m%d_%H%M%S)

mv TraffiDent_SupportBench TraffiDent_SupportBench_v121_backup_${STAMP}
mv data/processed/traffident/supportbench data/processed/traffident/supportbench_v121_backup_${STAMP}
mv data/diagnostics/traffident/supportbench data/diagnostics/traffident/supportbench_v121_backup_${STAMP}
[ ! -d artifacts/supportbench ] || mv artifacts/supportbench artifacts/supportbench_v121_backup_${STAMP}

tar -xzf TraffiDent_SupportBench_v1.2.2.tar.gz
mv TraffiDent_SupportBench_v1.2.2 TraffiDent_SupportBench
cd TraffiDent_SupportBench
chmod +x build_all.sh pbs/*.sh pbs/*.pbs derived/*.py tests/*.py
conda activate TFP
python preflight.py
python tests/run_dependency_lag_smoke.py
qsub pbs/build_supportbench.pbs
```

After the canonical build, inspect the Stage-06 `predictive_context_audit`. Then run Net06 candidate/headroom diagnostics before training Qwen.
