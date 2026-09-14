# Baseline v0.6.0

Baseline v0.6.0 is the clean rerun release aligned with Backbone v0.5.0 and ExperimentCommon v1.0.0. Baseline methods keep their existing architectures/vendor sources; the release consolidates shared data, metrics, prediction artifacts, and frozen-backbone consumption.

## Experiment groups

Main table:

- FrozenBackbone
- ImputeFormer
- GinAR
- RAFT
- RAST
- SARAF

Additional tracks:

- LEAF — LLM-specific comparison
- SLPF — fixed partial-sensing sub-track
- RAST-PatchSTG — controlled appendix adaptation

## Shared cache

Baseline no longer materializes or aliases a second copy of the benchmark. Every runner reads **directly** from:

```text
ExperimentCache/TraffiDent_Net_06/development
ExperimentCache/TraffiDent_Net_06/formal
ExperimentCache/TraffiDent_Net_06/evaluation
```

`Baseline_exps/` contains method/run artifacts only. No `Baseline_exps/_cache` directory or symlink is created or required.

Backbone, Baseline, and future OwnModel experiments therefore use exactly the same sample order, masks, TOD/DOW, scaling statistics, protocol hash, metric implementation, and prediction layout.

## Frozen PatchSTG policy

Baseline is a read-only consumer of `Backbone_exps/FROZEN_BACKBONE.json`.

- Baseline never scans Backbone runs to choose a checkpoint.
- Baseline never writes or overwrites the Backbone frozen manifest.
- ImputeFormer and RAST-PatchSTG use the same single PatchSTG reference checkpoint for Baseline seeds 10/20/30.

## Prepare data

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
conda activate TFP

python scripts/preflight.py --stage development
python scripts/prepare_data.py --stage development
python scripts/validate_cache.py --stage development
```

For formal:

```bash
python scripts/preflight.py --stage formal
python scripts/prepare_data.py --stage formal
python scripts/validate_cache.py --stage formal
python scripts/retrieval_leakage_audit.py --stage formal
```

If the shared formal cache was already built by Backbone v0.5.0, these commands validate and reuse it directly rather than rematerializing it.

Before methods that depend on PatchSTG:

```bash
python scripts/validate_frozen_patchstg.py
```

Then submit the main formal methods according to the supplied scripts. Freeze Baseline selection only after all required formal seeds are complete.
## Remove legacy `_cache` aliases

After deploying this direct-cache revision, old symlink trees can be removed safely with:

```bash
python scripts/cleanup_legacy_cache_aliases.py
```

The cleanup refuses to delete real files or non-empty real data directories.


## 2026-09-13 runtime hotfix: RAST + LEAF

- RAST keeps `vendor/RAST` frozen. The Baseline runner installs an import-only compatibility shim for RAST's eager optional STID/AGCRN imports. The shim raises immediately if native RAST unexpectedly tries to instantiate either optional pretrained backbone.
- RAST uses the existing TFP environment with `faiss-gpu`; run `bash scripts/setup_rast_faiss.sh` once if FAISS is missing.
- LEAF no longer requires `/envs/LEAF`. GraphBranch and HypergraphBranch run in the shared TFP environment through `baseline/runners/leaf_runner.py`.
- LEAF's LLM selector is intentionally post-hoc JSON. No vLLM installation is performed in TFP. A selector JSON contains one row-major query-node entry per element with `final_answer` equal to 1 (GraphBranch), 2 (HypergraphBranch), or null (defaults to GraphBranch).

Quick checks:

```bash
python tests/smoke_rast_leaf_hotfix.py
python scripts/check_rast_runtime.py
python scripts/check_leaf_tfp_runtime.py
```

Resubmit only the failed jobs:

```bash
bash scripts/resubmit_rast_development.sh
bash scripts/resubmit_leaf_development.sh
```

## RAST native-import compatibility r4

The external native RAST runner deliberately does not install EasyTorch/BasicTS training dependencies.  Frozen RAST eagerly imports optional pretrained-backbone helpers, which in turn import `basicts.utils`; importing the vendor `basicts` package would execute its launcher and require EasyTorch even though native RAST never uses that path.  The Baseline compatibility layer therefore injects guarded import-only placeholders for `baselines.STID`, `baselines.AGCRN`, and `basicts.utils` before importing frozen RAST.  If a pretrained STID/AGCRN path is ever requested, the placeholders fail loudly instead of silently changing the experiment.
