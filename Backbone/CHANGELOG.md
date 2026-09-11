# Changelog
- Hotfix: split `clone_or_keep` local declarations in `setup_official_models.sh` so `set -u` does not expand `$name` before assignment.

## v0.4.0 — 2026-09-10
- Added PatchSTG (KDD 2025), frozen at `8b298b230227a304a082e05241c69dca3740a73d`.
- Added DST-SGNN (IJCAI 2025), frozen at the pure numerical 2025 snapshot `ce426c7f0382aecb1a09c731fdc5ad411b36dfdc`.
- Added official-source-preserving adapters and output-shape contracts.
- Added AdamW, Adagrad, ExponentialLR and configurable scheduler stepping interval.
- Added Stage-1B `submit_recent_screening.sh`.
- Screening summary expanded from 4 to 6 candidates while preserving the original selection rule.
- Added screening-only backward compatibility with the frozen v0.3.1 cache/results; Formal/Evaluation require v0.4.0 caches.
- Added PatchSTG dependency `timm==1.0.12`.
- Hardened preflight imports and added VS Code/Pyright extra paths.
- Preserved prior BigST shape, RMSE, STDN warning and GPU metric-refresh hotfixes.

## 2026-09-11 DST-SGNN numerical-stability hotfix
- Keep pinned DST-SGNN vendor source unchanged.
- Wrap only DST-SGNN forward calls in an adapter-local stable `torch.linalg.eigh` context.
- Symmetrize the official Gram matrix and add a tiny relative diagonal ridge (`1e-6`); rare failures fall back to a stronger `1e-4` ridge in float64 before casting back.
- Adapter revision: `dstsgnn_official_h12_patch12_eigh_stable_v2`.
- Added `scripts/smoke_dstsgnn_eigh_hotfix.py`.
