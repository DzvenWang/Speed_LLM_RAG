# PreSuRe-P1

Minimal retrieval-sidecar experiment for **Frozen PatchSTG under incomplete sensing**.

P1 deliberately combines established components rather than inventing a new large architecture:

- PatchSTG supplies the strong frozen numerical forecast and pre-head representation.
- RAFT-style historical episode memory supplies causally completed historical support and its subsequent trajectory.
- A tiny zero-initialized forecast-residual adapter tests whether that support contains incremental predictive information.

## What P1 does NOT contain

No input filling, no region action oracle, no utility teacher, no Need module, no harm veto, no Gate, no Qwen/LLM, and no end-to-end PatchSTG fine-tuning.

## Required sibling workspace

Place `PreSuRe` under the existing `Speed_LLM_RAG` workspace with:

- `Backbone`
- `Backbone_exps`
- `ExperimentCache`
- `ExperimentCommon`
- `TraffiDent_SupportBench`
- `data/processed/traffident/supportbench`

On another layout set `PRESURE_WORKSPACE_ROOT`.

## Local Windows run

```powershell
conda activate TFP
cd "C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG\PreSuRe"
python tests\smoke_p1.py
python scripts\preflight_p1.py --stage development
python scripts\run_p1.py --stage development
```

or:

```powershell
.\run_p1.ps1 -Stage development
```

The first run prepares frozen PatchSTG artifacts and retrieval evidence. Later runs reuse them unless `--force-prepare` / `-ForcePrepare` is used.

## Gadi

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
qsub pbs/presure_p1_development.pbs
```

## Main outputs

Artifacts:

`PreSuRe_exps/_p1_episode_artifacts/TraffiDent_Net_06/development/`

Runs:

`PreSuRe_exps/PreSuRe_P1/TraffiDent_Net_06/development/<RUN_TAG>/`

Logs:

`PreSuRe_logs/p1/<RUN_TAG>/`

## Decision rule

Do not judge P1 only against PatchSTG. The important comparisons are:

1. `history_future` vs `adapter_only`
2. `history_future` vs `shuffled_support`
3. `history_future` vs `history_only`

If real retrieved episode evidence does not beat both capacity and random-support controls, do not add P2/P3 complexity yet.
