# Baseline v1.0.0 clean restructure

## Primary experiments

ImputeFormer, LRTC-TMCP, GinAR, RAFT, RAST, SARAF, ReMFT.

## Secondary / controlled experiments

SPRINT, SLPF, MoGERNN, LEAF, LEAF-Qwen1.5B, RAST-PatchSTG.

## Deliberately removed

- old-result project-version compatibility;
- result migration / legacy cache cleanup;
- method implementation revision gates;
- status-snapshot submission scripts tied to previous logs;
- duplicate Primary PBS entry points.

## New Primary execution path

`smoke_primary.pbs -> development_primary.pbs -> GinAR stability probe -> formal_primary.pbs -> freeze_formal_selection.py -> evaluation_main.pbs`.

## Validity fixes

- GinAR Formal divergence now exits non-zero before any selectable artifact is written.
- GinAR Formal uses LR=5e-4 and gradient clip=1.0 and must first pass an 8-epoch full-Formal-population stability probe.
- ReMFT removes runtime-only random frozen BERT layers when the notebook expands beyond the local pretrained checkpoint, then requires exact checkpoint/runtime BERT depth/hidden/head consistency.
- LRTC-TMCP remains strict query-local H12 and deterministic seed10-only.

## Gadi runtime checks still required

1. Run all Primary GPU smokes.
2. Confirm ReMFT prints `REMFT BERT DEPTH NORMALIZED ...` followed by an exact `REMFT LLM CONTRACT` pass.
3. Confirm GinAR stability probe reaches all configured probe epochs without `GINAR VALIDATION DIVERGENCE STOP`.
