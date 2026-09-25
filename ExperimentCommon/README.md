# ExperimentCommon v1.2.0

Shared materialization/metric contract for `Speed_LLM_RAG`, aligned with **Protocol v2.1.0** and TraffiDent-SupportBench v1.2.6 artifacts.

## Current experiment policy

- Training/development protocol: `protocol_core5_s40`
- Development: 12,000 train + 5,000 val, five families exactly balanced
- Formal: 50,000 train + 5,000 val, five families exactly balanced
- Frozen paired evaluation: `protocol_core5_s20`, `protocol_core5_s40`, `protocol_core5_s60`, `protocol_core5_s90`
- All four evaluation caches are **pre-materialized now** from the already-frozen Protocol manifests.
- Protocol evaluation access can remain LOCKED while caches are being prepared. This is a cache-preparation bypass only; ordinary model consumers must still respect the Protocol access gate before using Test tensors.

ExperimentCommon never resamples Protocol rows. It materializes exactly the frozen Protocol view manifests.

## Portable layout

The project root comes from `SPEED_LLM_RAG_ROOT` when set, otherwise from the normal sibling-project layout. No Windows/Gadi absolute path is required.

```text
Speed_LLM_RAG/
  ExperimentCommon/
  ExperimentCache/
  ExperimentCommon_logs/
  Protocol/
  Protocol_artifacts/
  TraffiDent_SupportBench/
```

## Collision-proof cache identity

```text
ExperimentCache/
  <dataset>/
    <protocol_name>/
      s<severity>/
        <stage>/
          <full_protocol_stage_hash>/
```

The cache manifest records protocol name/version, severity, paired group, stage hash, split view hashes, contract fingerprint, time-feature fingerprint, and whether an evaluation cache was materialized while the Protocol access gate was locked.

## Build everything now

```bash
cd ExperimentCommon
python scripts/build_caches.py --force
```

This builds and validates:

- `protocol_core5_s40/development`
- `protocol_core5_s40/formal`
- `protocol_core5_s20/evaluation`
- `protocol_core5_s40/evaluation`
- `protocol_core5_s60/evaluation`
- `protocol_core5_s90/evaluation`

If Protocol evaluation access is still locked, the four Test caches are still materialized, but their `READY.json`/`cache_manifest.json` record that fact.

To build training caches only:

```bash
python scripts/build_caches.py --training-only --force
```

## Evaluation governance

Pre-materializing Test tensors means they are physically present on disk. Therefore the evaluation lock is now a **program/procedure access lock**, not physical isolation.

Downstream Backbone/PreSuRe code should call `experiment_common.cache.require_evaluation_cache_access(cfg, 'evaluation')` before consuming an evaluation cache. Until `Protocol/scripts/grant_evaluation_access.py` is run, that guard raises `PermissionError`.

## Direct commands

Training cache:

```bash
python scripts/prepare_cache.py --stage formal --force
python scripts/validate_cache.py --stage formal
```

Locked evaluation cache preparation:

```bash
python scripts/prepare_cache.py --config configs/experiment_contract_eval_s20.yaml --stage evaluation --materialize-locked-evaluation --force
python scripts/validate_cache.py --config configs/experiment_contract_eval_s20.yaml --stage evaluation --validate-locked-evaluation
```
