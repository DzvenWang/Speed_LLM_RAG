# Migration: Baseline v0.3.1 -> v0.4.0

## What changed

- Main-table story was rebuilt around FrozenBackbone / ImputeFormer / GinAR / RAFT / RAST / SARAF.
- Added GinAR and SARAF official source governance.
- Added external Val-only runners for GinAR, RAFT and SARAF.
- Added Protocol-time overlap exclusion for retrieval baselines.
- LEAF moved to LLM-specific comparison; SLPF explicitly remains a fixed partial-sensing subtrack.
- Cache schema upgraded to v2: materialized data identity is separated from the set of baseline source repositories.

## What does not change

Shared Protocol, view identities, sensing masks, normalized/raw arrays, 12->12 task definition and Development 12k/5k views do not change.

## Existing Development cache

Do **not** rebuild 12k/5k just because two baselines were added. After backing up the Baseline project and installing v0.4.0:

```bash
bash scripts/setup_official_models.sh
bash scripts/strip_vendor_git.sh
python scripts/verify_official_sources.py
python scripts/migrate_cache_v040.py --stage development
python scripts/validate_cache.py --stage development
python scripts/preflight.py
python scripts/prepare_bridges.py development
qsub pbs/compatibility_audit.pbs
```

The migration script first verifies the old fingerprint and frozen Protocol rows, makes a backup of the old cache manifest, and changes only cache identity metadata. It does not rewrite `.npy` tensors or split manifests.
