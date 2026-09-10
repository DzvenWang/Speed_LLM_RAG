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
