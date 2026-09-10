# Shared Protocol v1.0.0

`Protocol` is the single source of truth for which TraffiDent-SupportBench sensing views are used by Backbone, Baseline, and the later proposed model.

## Fixed experiment definition

- Dataset: `TraffiDent_Net_06`, N=388
- History/Horizon: 12/12 (5-minute steps)
- Severity: 0.40, distribution: ID
- Families: Temporal, Regional, Directional, Predictive Regional, Predictive Directional
- Development: Train 12,000 = 2,400×5; Val 5,000 = 1,000×5
- Formal: all eligible unique Train view-groups; exact/near-exact five-family balance; **same Val views as Development**
- Evaluation: Test 5,000 = 1,000×5, generated only after explicit authorization

Every selected forecast target has exactly one `view_group_id` occurrence inside a split. Family assignment and concrete `view_id` are frozen in CSV manifests.

## Artifact layout

`/scratch/mh20/zw1165/Speed_LLM_RAG/Protocol_artifacts/TraffiDent_Net_06/protocol_v1/`

Each stage contains `*_views.csv`, `stage_manifest.json`, `FROZEN.json`, and frozen Val/Test evaluation masks.

## First run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Protocol
conda activate TFP
python scripts/preflight.py
qsub -v STAGE=development pbs/build_protocol.pbs
```

After the development job exits with `SCRIPT_EXIT_STATUS=0`:

```bash
python scripts/status.py
python scripts/validate_protocol.py --stage development
```

Backbone screening and Baseline development may then be materialized. Build `formal` only when the project reaches the full-train stage:

```bash
qsub -v STAGE=formal pbs/build_protocol.pbs
```

Do **not** build Test now. After all method configurations are frozen:

```bash
python scripts/authorize_evaluation.py --acknowledge-all-methods-frozen
qsub pbs/build_evaluation_protocol.pbs
```

## Consumer rule

Backbone/Baseline may transform Protocol views into method-specific tensors, but they are forbidden to resample, reassign sensing family, or change query identity.
