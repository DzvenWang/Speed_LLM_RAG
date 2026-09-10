# Changelog

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
