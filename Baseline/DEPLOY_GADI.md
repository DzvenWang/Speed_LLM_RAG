# Deploy Baseline v1.1.0 Core-5 on Gadi

Prerequisites already prepared by the shared benchmark pipeline:

- Protocol v2.1.0 Core-5 artifacts
- ExperimentCommon v1.2.0 Core-5 caches
- Backbone v1.0.10 `FROZEN_BACKBONE.json`

Deploy the new Baseline directory, then:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
conda activate TFP
bash scripts/setup_official_models.sh
bash scripts/setup_remft_bert.sh
python scripts/verify_official_sources.py
python tests/smoke_core5_contract.py
python scripts/baseline_status.py
```

Prepare/validate the Core-5 s40 training cache:

```bash
qsub pbs/prepare_formal.pbs
```

Run development and formal experiments using the normal Primary-7 submission scripts. Old Baseline v1.0.0 results are not selectable under v1.1.0.

Each Primary Formal job now performs its own evaluation immediately after successful training. The exact completed Formal run is used directly; no Baseline-level selection/freeze artifact is required.

```bash
bash scripts/submit_primary_formal_seed10.sh
```

Each job trains on Core-5 s40, then evaluates that run's `best.pt` on s20, s40, s60 and s90 and prints final severity/scenario tables in the same live log. After the jobs finish:

```bash
python scripts/primary_run_status.py
```

Optional cache-only validation remains available with `qsub pbs/prepare_evaluation.pbs`.
