# Base_SupportBench v1.2.0

Unified preprocessing for Backbone/Baseline. Supports TraffiDent_Net_01/02/03/04/06 and target `speed` or `flow`.

## Contract
- Raw channels are resolved from `channel_mapping.json` (fallback: flow, occupancy, speed).
- Five incomplete-sensing families and 20/40/60/90 severity generation keep the previous SupportBench General-track configuration.
- Training/validation protocol uses severity 40%; evaluation uses paired 20/40/60/90 test queries.
- Speed and Flow artifacts are physically isolated under target-specific benchmark, protocol, cache, log and experiment paths.
- No RAG/support-memory/challenge-pool/predictive-sensing derived artifacts are built.

## Net-06
```powershell
python -u Base_SupportBench\scripts\preflight.py --dataset TraffiDent_Net_06 --target speed
python -u Base_SupportBench\scripts\build_dataset.py --dataset TraffiDent_Net_06 --target speed --force

python -u Base_SupportBench\scripts\preflight.py --dataset TraffiDent_Net_06 --target flow
python -u Base_SupportBench\scripts\build_dataset.py --dataset TraffiDent_Net_06 --target flow --force
```

Generated caches are under `ExperimentCache/<target>/...`; resolved consumer configs are under `Base_SupportBench_artifacts/resolved_configs/<target>/<dataset>/`.

## Consumers
Set `TRAFFIDENT_TARGET=speed` or `flow` (PBS wrappers also accept `TARGET=speed|flow`). Backbone and Baseline must use checkpoints/results from the same target identity.

## Portable Speed / Flow workflow

This project is the only preprocessing layer required by Backbone and Baseline.
Set `SPEED_LLM_RAG_ROOT` to the repository root (optional when the standard sibling layout is used), then choose `--target speed` or `--target flow`.
The canonical channel mapping is flow=0, occupancy=1, speed=2. Speed and Flow use separate benchmark, protocol, cache and log namespaces. The five general sensing families and 20/40/60/90 severity generation retain the original SupportBench sensing contract. Training/validation is selected at 40%; evaluation is paired across 20/40/60/90.
Generated resolved YAML files persist `${SPEED_LLM_RAG_ROOT}` tokens rather than machine-specific absolute paths, so artifacts can move between Windows and Gadi.
