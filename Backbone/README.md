# Backbone v0.5.0

Backbone v0.5.0 is the clean rerun release for the TraffiDent incomplete-sensing benchmark. It keeps the five numerical architectures and pinned official sources unchanged, while moving all cross-project experiment semantics into **ExperimentCommon v1.0.0**.

## Active models

- STAEformer
- HimNet
- BigST
- STDN
- PatchSTG

Five-model screening remains the comparison/verification experiment. `PatchSTG` is explicitly configured as the downstream numerical reference model and is trained on the formal stage with seeds 10/20/30. Seed 10 is the fixed downstream reference checkpoint after the three-seed formal run is summarized.

## Shared Experiment Contract

Backbone no longer owns a private materialized data cache. `scripts/prepare_data.py`, training, summaries, and evaluation all read **directly** from the canonical ExperimentCache:

```text
ExperimentCache/TraffiDent_Net_06/development  <- Backbone screening
ExperimentCache/TraffiDent_Net_06/formal       <- Backbone formal
ExperimentCache/TraffiDent_Net_06/evaluation   <- test after freeze
```

`Backbone_exps/` contains model/run artifacts only. No `Backbone_exps/_cache` directory or symlink is created or required.

The common contract fixes:

- sample membership from frozen Shared Protocol;
- one TOD/DOW resolver and one temporal fingerprint;
- one cache/data identity;
- one point-level micro MAE/RMSE implementation;
- one prediction NPZ contract `[M, P, N]`.

Per-query metrics are still written for diagnostics, but model selection and reported aggregate metrics use the common point-level evaluator.

## Recommended rerun

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
conda activate TFP

python scripts/preflight.py --stage screening
python scripts/prepare_data.py --stage screening --force
python scripts/validate_cache.py --stage screening
bash scripts/submit_screening.sh
```

After all five screening runs finish:

```bash
python scripts/summarize_screening.py
```

Prepare and run the configured reference model formally:

```bash
python scripts/preflight.py --model PatchSTG --stage formal
python scripts/prepare_data.py --stage formal
python scripts/validate_cache.py --stage formal
bash scripts/submit_reference_formal.sh
```

After PatchSTG seeds 10/20/30 finish:

```bash
python scripts/summarize_formal.py
```

This creates the Backbone-owned `Backbone_exps/FROZEN_BACKBONE.json`. Baseline and future OwnModel projects are read-only consumers of that artifact.

## Important

Do not replace the project code underneath already-running v0.4.1 jobs and then combine those runs with v0.5.0 results. v0.5.0 intentionally changes the common cache identity and aggregate metric semantics, so its formal evidence should be rerun under this release.
## Remove legacy `_cache` aliases

After deploying this direct-cache revision, old symlink trees can be removed safely with:

```bash
python scripts/cleanup_legacy_cache_aliases.py
```

The cleanup refuses to delete real files or non-empty real data directories.

