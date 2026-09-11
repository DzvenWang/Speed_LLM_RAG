# Changelog

## v0.4.0 — Story-Aligned Baselines
- Added GinAR (KDD 2024) as the missing-aware end-to-end main-table baseline.
- Added SARAF (KDD 2026) as the retrieval-reliability main-table baseline.
- Implemented external Train/Val-only runners for GinAR, RAFT and SARAF.
- RAFT/SARAF use Train-only historical memory and explicit Protocol-time interval overlap exclusion.
- Reorganized LEAF to LLM-specific comparison and SLPF to fixed partial-sensing subtrack.
- Added cache schema v2 and safe v0.3.1 Development cache migration without tensor rebuild.
- Added source-lock `resolve_once` policy for repositories whose exact commit is first resolved on Gadi.
- Added method-specific GPU smoke and guarded Development submission.
- Test remains physically isolated.

## 2026-09-11 hotfix
- Save prediction `view_id` as pickle-free NumPy Unicode arrays instead of `dtype=object`.
- Add `scripts/recover_incomplete_result.py` so completed GinAR/SARAF training runs can be finalized without retraining.
- Move RAFT precomputed retrieval tensors onto the model device before indexed training/validation access.
- Harden 5-minute timestamp parsing: numeric row indices are no longer interpreted as nanosecond datetimes.
- Add SARAF time-bonus probe; constant time-aware bonuses now fail loudly instead of silently producing an unfair baseline.
- Copy mmap-backed dataset slices before Torch collation to remove non-writable-array warnings.
