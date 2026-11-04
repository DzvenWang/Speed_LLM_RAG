# PCFT-STLA v6.0.0 + Propagation Identifiability Diagnostic v1.3 — Unified Source

This archive contains the **entire original PCFT v6.0.0 source tree** and the
**entire latest propagation identifiability diagnostic v1.3 source tree**.
It does not replace v6 with the diagnostic and does not read old PCFT checkpoints
or caches. All original v6 files are kept unchanged.

## Contents

- `pcft_stla/v6/` — original v6 physical/functional graph, data, model, training, pipeline, utilities.
- `configs/pcft_v6.yaml` — original v6 config (unaltered).
- `pbs/formal_v6.pbs`, `scripts/run_v6.py`, `scripts/preflight_v6.py` — original formal run.
- `tests/` — original 34 v6 tests.
- `diagnostics/propagation_identifiability/` — complete v1.3 statistical direction/lag + STLA incremental-gain diagnostic, GPU/CPU PBS, scripts, 16 tests.
- `scripts/verify_unified_package.sh` — syntax, dependency, storage, test checks with optional CPU real-data preflights.
- `SOURCE_MANIFEST_SHA256.txt` — SHA-256 hashes for all shipped source files.

## Existing siblings required on Gadi (not in this source archive)

The project expects these **existing** sibling resources under `Speed_LLM_RAG`:
`Backbone/` (with official STLAformer vendor source), `Base_SupportBench/`,
processed TraffiDent data, formal Backbone caches, and original STLA best.pt
checkpoints. **No large datasets, official vendor code, model weights, or
experiment results are bundled.**

## Install / merge

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
unzip -oq PCFT-STLA_v6.0.0_plus_PropagationDiagnostic_v1.3_Complete_Source.zip
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/PCFT-STLA:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
bash PCFT-STLA/scripts/verify_unified_package.sh
```

The ZIP contains just `PCFT-STLA/...`: extracting it does **not** delete other
PCFT files, sibling projects, or existing run outputs. If you want to update
files in-place, make sure no queued PBS job is reading them; PBS scripts can be
snapshotted by the scheduler at submission, while imported Python files are read
at job runtime.

Run the **CPU, real-data preflight** before submitting another GPU job:

```bash
bash PCFT-STLA/scripts/verify_unified_package.sh --preflight
```

The diagnostic-specific checker is unchanged and can also be run directly:

```bash
bash PCFT-STLA/diagnostics/propagation_identifiability/scripts/pre_submit_check.sh
```

## GPU jobs (only after both real-data preflights pass)

Diagnostic Speed:

```bash
qsub -N PGain_S -v TARGET=speed,SEED=10,CHECKPOINT=Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt PCFT-STLA/diagnostics/propagation_identifiability/pbs/propagation_diagnostic.pbs
```

Diagnostic Flow:

```bash
qsub -N PGain_F -v TARGET=flow,SEED=10,CHECKPOINT=Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt PCFT-STLA/diagnostics/propagation_identifiability/pbs/propagation_diagnostic.pbs
```

Original v6 formal Speed and Flow jobs use `PCFT-STLA/pbs/formal_v6.pbs` and
`BACKBONE_CHECKPOINT` instead of the diagnostic's `CHECKPOINT` env var. See the
original `README.md` for exact commands.

## Verification boundaries

Bundling source and passing offline tests does **not** verify Gadi's mounted
storage, specific cached TRAIN/VAL files, or actual V100 CUDA execution.
Always run `--preflight` on Gadi before submitting jobs. The diagnostic
produces a retrospective statistical / predictive association analysis;
its edge counts are **not** physical-causality confirmations.

Original v6 config metadata contains `version: 5.0.3` despite having v6 source.
It is preserved unchanged in this exact-source merge rather than silently
modifying the training configuration or graph/cache identities.

## v7.0 unified architecture (new)
See [README_V7.md](README_V7.md) for STLA Anchor, C1, C2, C3, TRAIN-only utility learning, mixed-mask optional input, GPU smoke, Gadi PBS and resume. The original v6 source remains unchanged.
