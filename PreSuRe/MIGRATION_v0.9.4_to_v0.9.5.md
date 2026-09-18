# PreSuRe v0.9.4 -> v0.9.5

v0.9.4 established a representation bottleneck: the query-only direction probe achieved only weak masked direction discrimination (q75 AUROC ~0.556, q90 AUROC ~0.580; signed-error Spearman ~0.058). v0.9.5 therefore does not add another retrieval/acceptance policy. It tests whether a richer **Query Error-State Representation** can recover the missing future-correction signal.

## New query-side evidence

1. **Frozen PatchSTG predictive hidden state.** A forward-hook discovery pass selects a deep node-aligned latent activation from the exact frozen PatchSTG checkpoint. The chosen module name, raw shape/width and deterministic projection are written to `query_error_state_artifact_manifest.json` for auditability. No backbone parameter is updated.
2. **Strictly causal recent backbone-error state.** Previous-query PatchSTG residuals are used only after their forecast timestamps are already realized at the current query time. Chronology is derived from an absolute cache-manifest anchor when available; otherwise TOD/DOW row order must pass a conservative proof. If chronology cannot be established, this channel is disabled and filled with zeros rather than guessing.
3. **Physical transition state.** Query-history-only descriptors encode local slope/acceleration/shock, neighbour gradient and sensing continuity.

## Controlled ablations

The probe trains `base`, `hidden`, `recent_error`, `physical`, `hidden_physical`, and `full`. The representation variant is selected **only on Train-dev**. Final Validation is evaluation-only and cannot choose a variant.

## Leakage contract

Current-query future is used only as Train supervision and diagnostic evaluation. Retrieval candidates, support residuals, Oracle choices, Qwen, Dual Memory, direct/support evaluation masks and current-query future targets are not representation inputs.

## Run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_query_error_state_probe.py
python scripts/prepare_query_error_state_artifacts.py --stage development
python scripts/query_error_state_preflight.py --stage development
bash scripts/submit_query_error_state_probe.sh
```

The PBS submission also prepares missing/current-contract v0.9.5 artifacts automatically.
