# Backbone v1.2.0

Backbone training/evaluation project for TraffiDent incomplete-sensing experiments.

## Models
STAEformer, HimNet, BigST, STDN, PatchSTG, STLAformer.

## Protocol
- targets: `speed`, `flow`
- Formal seed: `10`
- Formal train/validation: s40
- automatic evaluation: s20 / s40 / s60 / s90

## Checkpoint lifecycle
There is **no Backbone freeze/registration stage**.

A successful Formal run produces its own `best.pt` and, after automatic multi-severity evaluation succeeds, updates model-local convenience pointers:

```text
Backbone_exps/<target>/<model>/TraffiDent_Net_06/formal/LATEST_RUN.txt
Backbone_exps/<target>/<model>/TraffiDent_Net_06/formal/LATEST_CHECKPOINT.txt
```

These pointers are not model selection artifacts. Downstream experiments explicitly specify the Backbone model they want to use and may optionally provide an exact checkpoint path.

## Local smoke
```powershell
python -u Backbone\scripts\smoke_all_formal.py --continue-on-error
```

## Gadi Formal
```bash
qsub -v MODEL=STLAformer,TARGET=speed,SEED=10 -N STLA_speed Backbone/pbs/run_model_formal.pbs
qsub -v MODEL=STLAformer,TARGET=flow,SEED=10  -N STLA_flow  Backbone/pbs/run_model_formal.pbs
```

Runtime chain:

```text
preflight -> train -> best.pt -> evaluate s20/s40/s60/s90 -> publish latest model-local pointer
```


## Portable latest pointers (v1.2.1)
`LATEST_RUN.txt` and `LATEST_CHECKPOINT.txt` store paths relative to `SPEED_LLM_RAG_ROOT`, using `/` separators, for example:

```text
Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10
Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt
```

The same pointer files can therefore be copied between Windows and Gadi. Legacy absolute pointers are still accepted and rebased when possible. Existing runs can be migrated with `python -u Backbone/scripts/refresh_latest_pointers.py`.
