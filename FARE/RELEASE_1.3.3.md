# FARE v1.3.3 — Gadi PBS `BASHRCSOURCED` fix

## Failure confirmed from actual Gadi PBS output

Both Net-06 Pilot jobs, Speed `180808027` and Flow `180808028`, stopped in approximately 21–22 seconds with `/etc/bashrc: line 12: BASHRCSOURCED: unbound variable`. Exit status 1, GPU utilization 0%, GPU memory 0 B. This occurred before model loading or training.

## Root cause and correction

In v1.3.2 `pbs/_activate_tfp.sh` started by sourcing `~/.bashrc` while its caller `train_formal.pbs` used `set -euo pipefail`. A Gadi shell startup script reads `BASHRCSOURCED` without defaulting it; Bash nounset causes an immediate exit even though the original shell script used `source ~/.bashrc || true`.

The new `_activate_tfp.sh` NEVER reads `~/.bashrc` or `/etc/bashrc`. It instead sources an explicit `conda.sh` from a short, trusted list (`FARE_CONDA_SH`, `../miniconda`, `../miniconda3`, `$HOME/miniconda`, `$HOME/miniconda3`). It temporarily disables Bash nounset only around Conda sourcing/activation, restores it afterward, and checks `CONDA_DEFAULT_ENV=TFP`. Errors emit actionable `FARE_CONDA_INIT_ERROR` or `FARE_CONDA_ACTIVATE_ERROR`.

All four PBS scripts (`train_formal.pbs`, `evaluate.pbs`, `oof_controller.pbs`, `calibrate.pbs`) use the same bootstrapping path. If `FARE_CONDA_SH` is set in the login shell, `submit_gadi.py` forwards it with `qsub -v` for every job. Neither Python model, train configuration, checkpoint format, nor in-GPU Train→Val→Test scheduling changes.

## Verify on Gadi login node before submitting

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
export SPEED_LLM_RAG_ROOT="$PWD"
bash --noprofile --norc -c 'set -euo pipefail; source FARE/pbs/_activate_tfp.sh; python -c "import torch; print(torch.__version__)"'
```

If needed: `export FARE_CONDA_SH=/scratch/mh20/zw1165/miniconda/etc/profile.d/conda.sh` before local verification. The normal job autodiscovery already considers that path.

## Resubmit Net-06 Pilot with one GPU PBS job per target

```bash
python -u FARE/scripts/submit_gadi.py train \
  --dataset TraffiDent_Net_06 \
  --targets speed flow --seed 10 \
  --config FARE/configs/pilot.yaml \
  --eval-splits val test --severities 20 40 60 90
```

The old failed jobs are not restartable. A successful submission alone does not prove node-side execution; check the real PBS output for `FARE_CONDA_READY` and `FARE_GADI_TRAIN_START`.

## Test coverage

- A deliberately hostile `.bashrc` reproduces `BASHRCSOURCED` when sourced; all four PBS wrappers bypass it and execute a fake workload.
- Conda auto-discovery, override, missing-Conda failure and shell `-euo pipefail` are tested.
- Existing fake qsub and Train→same-allocation Evaluate tests retained.
