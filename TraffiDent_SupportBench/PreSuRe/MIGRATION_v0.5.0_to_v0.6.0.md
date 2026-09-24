# Migration v0.5.0 -> v0.6.0

v0.6.0 is a scientific supervision update.

## Why

v0.5 showed:
- Oracle-E@24 retained large node-local headroom;
- M3 Top-4 retained large node and node×horizon headroom;
- router eligibility removed almost none of that headroom;
- M4 learned execution still recovered only a tiny fraction;
- hard teacher pretraining raised support mass to ~0.87 without material forecasting gain;
- M5 binary gate supervision produced weak discrimination.

## Changes

1. M4 hard best-support CE -> soft local utility distillation.
2. Separate support intervention mass from conditional candidate weights.
3. Supervise candidate transported residual and final mixture correction directly.
4. M5 binary BCE -> continuous optimal intervention-strength target.
5. Add SoftMix@M3Top4 diagnostic.
6. Add gate AUROC / teacher MAE / positive-negative gate means / activated harm metrics.
7. Fix read-only NumPy -> Torch warning in local diagnostics.

M4 and M5 v0.5 checkpoints are intentionally incompatible. Rerun M4 and M5 under v0.6. Full Development submission reruns the complete single-GPU pipeline for clean comparability.
