# Migration: PreSuRe v0.9.5 → v0.9.5.1

v0.9.5.1 does not replace the v0.9.5 query-error representation experiment. It adds a stricter diagnostic layer that completes two tests left unresolved by v0.9.5.

## Why this release exists

The v0.9.5 log selected `input_st_fc` as the automatic PatchSTG hidden layer and disabled recent-error evidence because chronology could not be proven from row order. Therefore the weak v0.9.5 gain could not establish whether a deeper predictive latent state or a causal recent backbone-bias state was useful.

## New experiment

`QueryErrorStateRepresentationDiagnostics`:

1. Hook all PatchSTG modules, including parent blocks.
2. Catalogue every node-aligned activation without consulting Train-dev or Final Validation labels.
3. Screen up to 10 candidate layers using the exact v0.9A Train/Train-dev split.
4. Confirm the best two layers over seeds 10/20/30.
5. Repair recent-error chronology from manifest `window_start` (or an explicit absolute anchor if available).
6. Construct Validation recent-error features from Train forecast outcomes only.
7. Compare the selected hidden layer with physical and repaired recent-error representations over seeds 10/20/30.
8. Freeze layer, representation, and representative seed before Final Validation.

## New commands

```bash
python tests/smoke_query_error_state_representation_diagnostics.py
python scripts/prepare_query_error_state_representation_diagnostics.py --stage development
python scripts/query_error_state_representation_diagnostics_preflight.py --stage development
bash scripts/submit_query_error_state_representation_diagnostics.sh
```

## Leakage contract

- Current-query future is never a model input.
- Hidden-layer discovery uses only current incomplete query forward activations.
- Layer and representation selection use Train-dev only.
- Train recent-error features use earlier Train outcomes only after their targets are realised.
- Validation recent-error features use Train outcomes only; no Validation labels are recycled as inputs.
- Retrieval candidates, Oracle choices, Qwen, and Dual Memory remain excluded from this diagnostic.
