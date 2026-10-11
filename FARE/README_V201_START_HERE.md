# FARE v2.0.1 — Start Here

This archive contains the **entire standalone FARE project**, based on v2.0.0 and optimized for Gadi V100. It is not a diff/patch; replace only the `FARE` project folder after backing up the old one.

## Gadi sanity check

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
python -u FARE/scripts/preflight.py --target speed
python -u FARE/scripts/preflight.py --target flow
python -u FARE/scripts/train.py --target speed --seed 10 --smoke
python -u FARE/scripts/train.py --target flow --seed 10 --smoke
```

This project requires the **unchanged sibling projects** `Base_SupportBench/` and `Backbone/`, their formal cache artifacts and pretrained STLAformer checkpoints. It does not use archived `STLA_DASR` or `Support_Oracle` code.

## Formal jobs

```bash
qsub -N FARE_v201_speed -v TARGET=speed,SEED=10 FARE/pbs/train_formal.pbs
qsub -N FARE_v201_flow -v TARGET=flow,SEED=10 FARE/pbs/train_formal.pbs
```

By default: batch size 16, DataLoader workers 2, CPU OpenMP threads 3, memory 50k, coarse Top-64 → final Top-8, training for up to 115 epochs, one V100, 12 CPUs, 32GB RAM, 48-hour PBS walltime. Set `BATCH_SIZE=8` or `BATCH_SIZE=32` in `qsub -v` to benchmark. If batch 16 causes OOM in formal training, retry with batch 8.

## Live logs / progress

```bash
find FARE_logs/speed/TraffiDent_Net_06 -name '*.train.live.log' -printf '%T@ %p\n' | sort -nr | head -1
```

Read `FARE_EPOCH` for `train_seconds`, `validation_seconds` and `memory_refresh_seconds`. These are measured timings, not predicted overall runtime. Stage-specific evaluation and formal post-training evaluation remain enabled.

## Repeat original batch size

```bash
qsub -N FARE_v201_speed_b8 \
  -v TARGET=speed,SEED=10,BATCH_SIZE=8,NUM_WORKERS=0 \
  FARE/pbs/train_formal.pbs
```

See `RELEASE_2.0.1.md` for detailed source changes, precision comparisons, test coverage and limits.
