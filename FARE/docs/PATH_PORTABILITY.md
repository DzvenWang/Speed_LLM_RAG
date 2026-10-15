# FARE v4.0.1: portable workspace and Conda environment

## What is derived automatically

- Workspace: parent directory of this installed `FARE/` folder.
- Shared source: `<workspace>/Base_SupportBench` and `<workspace>/Backbone`.
- Shared resolved configs: `<workspace>/Base_SupportBench_artifacts/resolved_configs`.
- Backbone experiments: `<workspace>/Backbone_exps`.
- FARE output: `<workspace>/FARE_exps`, `<workspace>/FARE_logs`, `<workspace>/FARE_cache`.
- Submission Python interpreter: `sys.executable` and `sys.prefix` captured from the currently active environment.

No literal NCI username, scratch mount, gdata mount, or environment prefix is
encoded in the source code or PBS shell body. PBS is given absolute *runtime*
values by `submit_gadi.py`, as the compute node must be able to locate the
workspace and the exact Python executable across separate shells.

## Why this is necessary

A compute node does not necessarily know which `TFP` Conda environment was
activated at submission time. Activating an environment by name might select
an older identically named copy. The new PBS launcher does not call
`conda activate TFP`: it invokes the specific Python executable that submitted
the job and checks `sys.prefix` against the submitted `FARE_ENV_PREFIX`.

## Run from repository root

```bash
conda activate TFP
python -c 'import sys; print(sys.executable); print(sys.prefix)'
python -u FARE/scripts/preflight.py --target speed
python -u FARE/scripts/submit_gadi.py --target speed --seed 10
python -u FARE/scripts/submit_gadi.py --target flow --seed 10
```

Explicit checkpoints are optional (automatic LATEST/most recent checkpoint
resolution), but recommended for reproducible experiments:

```bash
python -u FARE/scripts/submit_gadi.py --target speed --seed 10 \
 --checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/<checkpoint-run>/best.pt
```

Use `--run-dir`, `--memory`, `--resume`, `--oos-forecast-file` with paths
relative to the repository root or absolute paths; both are supported.

## From a different current working directory

Invoke the installed FARE script by its path. Configuration defaults to the
installed FARE package, and `--config FARE/configs/pilot.yaml` or
`--config configs/pilot.yaml` is resolved from the repository root/package.

## One-job behavior

Each target has one PBS job, one V100, 12 CPUs, 32GB RAM, 48h max, as before.
Training validation is retained; after training, the default evaluation is
Test s20/s40/s60/s90, with `--eval-val` as an option.

## Known relocation caveat

A moved Conda environment does not affect old FARE experiment files because
those files are under the workspace, not in the Conda environment. Moving the
workspace itself *after* generating checkpoints may leave absolute references
in saved `metadata` and Base_SupportBench resolved config files. Those files
need migration or regeneration; this path patch does not silently reinterpret
old provenance paths.
