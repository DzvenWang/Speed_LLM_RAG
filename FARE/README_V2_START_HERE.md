# FARE 2.0 Quickstart

1. Read `RELEASE_2.0.0.md` for implemented features and unverified limitations.
2. Keep this folder under `Speed_LLM_RAG/FARE/` next to `Base_SupportBench/` and `Backbone/`.
3. Run `conda activate TFP` and `python -u FARE/scripts/preflight.py --target speed`.
4. Run pilot smoke: `python -u FARE/scripts/train.py --target speed --seed 10 --smoke`.
5. Submit formal: `qsub -N FARE_v2_speed -v TARGET=speed,SEED=10 FARE/pbs/train_formal.pbs`; repeat for flow.
6. Watch timestamped logs under `FARE_logs/<target>/<dataset>/` and metrics under `FARE_exps/`.

This is source code, not a trained checkpoint. Old v1.3.4 checkpoints and old memory archives cannot be treated as v2 checkpoint files.
