# Backbone v1.3.0 — incomplete vs. complete sensing for training

No changes to `Base_SupportBench`, `Baseline`, cache schema, model architectures, fixed seeds, frozen view selection, training recipes, or evaluation metrics. Applies to all six Backbone models (`STLAformer` recommended for the next experiment).

## Semantics

| | `incomplete` (default, old behaviour) | `complete` (new) |
|---|---|---|
| Formal Train | `train_x_obs_norm.npy` (s40 artificial sensing failures) | `train_x_full_norm.npy` (the **same** frozen train views with artificial failures removed) |
| Formal Val | `val_x_obs_norm.npy` | `val_x_full_norm.npy` |
| Raw-invalid historical observations | Keep normalized fill=0 | Keep normalized fill=0 using `history_observed OR artificial_missing` as the original valid mask |
| Target and loss | **Same** `*_y_raw.npy`, `*_future_valid.npy` | **Same** |
| Test/Evaluate | `test_x_obs_norm.npy`, s20/s40/s60/s90 | `test_x_obs_norm.npy`, s20/s40/s60/s90 |
| Checkpoint selection | Lowest incomplete-validation MAE | Lowest complete-validation MAE |
| Training run directory | `formal/` | `formal_complete/` |
| Training smoke directory | `formal_smoke/` | `formal_complete_smoke/` |

The training set size is unchanged: in the supplied Formal protocol it remains **50,000 train / 5,000 validation view rows**. `complete` is about *complete historical sensing*, **not** training on 100% of timestamps or using future observations. The matched frozen train/validation view rows and normalisation are deliberately unchanged so effects reflect the training input. The training-side artificial mask can still be present in the metadata; it does not mask `x` in complete mode. Therefore `val_masked_MAE` in complete-validation logs refers to would-have-been-masked locations, not masked inputs; the actual robustness comparison should use the identical s20/s40/s60/s90 Test caches.

**Important:** Complete inputs are only *complete relative to synthetic sensor failures*. Raw source invalidity is **never** recovered or replaced with ground truth. These positions retain the original model-fill value (normalised zero). `y_raw`, future validity, and evaluation metrics are never changed.

## Local commands — from `Speed_LLM_RAG` repo root

```powershell
# Preflight for the new complete train-input contract
python -u Backbone\scripts\preflight.py --stage formal --train-data-mode complete

# STLAformer bounded real-data smoke with automatic *smoke evaluation* s20/s40/s60/s90
python -u Backbone\scripts\smoke_all_formal.py --models STLAformer --targets speed --train-data-mode complete
python -u Backbone\scripts\smoke_all_formal.py --models STLAformer --targets flow --train-data-mode complete

# Formal full-data training; automatically evaluates four existing missing-test severities
python -u Backbone\scripts\run_model_formal.py --model STLAformer --target speed --seed 10 --train-data-mode complete
python -u Backbone\scripts\run_model_formal.py --model STLAformer --target flow  --seed 10 --train-data-mode complete

# Unchanged original 40%-missing training (equivalent to omitting --train-data-mode)
python -u Backbone\scripts\run_model_formal.py --model STLAformer --target speed --seed 10 --train-data-mode incomplete
python -u Backbone\scripts\run_model_formal.py --model STLAformer --target flow  --seed 10 --train-data-mode incomplete

# Inspect both pointers
python -u Backbone\scripts\status.py
```

## Gadi — full Formal training

All jobs keep the original `gpuvolta` queue and resources (`1x V100`, `12 CPUs`, `32GB`, `48h`). Mode passed through environment variable `TRAIN_DATA_MODE`; **default remains `incomplete`**.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG

qsub -N STLA_full_speed -v MODEL=STLAformer,TARGET=speed,SEED=10,TRAIN_DATA_MODE=complete Backbone/pbs/run_model_formal.pbs
qsub -N STLA_full_flow  -v MODEL=STLAformer,TARGET=flow,SEED=10,TRAIN_DATA_MODE=complete Backbone/pbs/run_model_formal.pbs

# Existing mode remains supported, or omit TRAIN_DATA_MODE:
qsub -N STLA_inc_speed -v MODEL=STLAformer,TARGET=speed,SEED=10,TRAIN_DATA_MODE=incomplete Backbone/pbs/run_model_formal.pbs
qsub -N STLA_inc_flow  -v MODEL=STLAformer,TARGET=flow,SEED=10,TRAIN_DATA_MODE=incomplete Backbone/pbs/run_model_formal.pbs
```

Successful formal jobs automatically follow `preflight -> training -> best.pt -> incomplete-test evaluation s20/s40/s60/s90 -> latest pointers`.

## Checkpoints and results

```text
Backbone_exps/<target>/STLAformer/TraffiDent_Net_06/
  formal/<run>_seed10/best.pt                   # missing-train (legacy path)
  formal/LATEST_CHECKPOINT.txt                # missing-train (legacy pointer)
  formal_complete/<run>_seed10/best.pt         # complete-train
  formal_complete/LATEST_CHECKPOINT.txt       # complete-train pointer
  checkpoint_evaluation/<run>/combined/...     # legacy missing-trained eval outputs
  checkpoint_evaluation/complete/<run>/combined/... # complete-trained eval outputs
```

The `best.pt`, `summary.json`, `RUN_MANIFEST.json`, and evaluation CSV/JSON include `train_data_mode`. Old checkpoints without this property are interpreted as `incomplete`. Both training modes are evaluated on exactly the *same missing input caches*, so their reported s20/s40/s60/s90 test errors can be directly compared, subject to identical model architecture and training recipe.

Full-trained checkpoint downstream: pass the **explicit** `formal_complete` checkpoint path to the dependent code, rather than the legacy Formal latest pointer. Existing Baseline code reading `formal/LATEST_CHECKPOINT.txt` remains unchanged.

## Re-evaluate an explicit checkpoint

```powershell
# Replace path below with actual produced best.pt
$env:TRAFFIDENT_TARGET = "speed"
python -u Backbone\scripts\run_checkpoint_evaluations.py --model STLAformer --checkpoint "Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal_complete/<run>_seed10/best.pt" --expected-seed 10 --severities 20 40 60 90
```

## Verification

```bash
python -m compileall -q Backbone
python -m unittest discover -s Backbone/tests -v
bash -n Backbone/pbs/run_model_formal.pbs Backbone/scripts/submit_formal.sh
```

Synthetic tests validate both input modes, raw-invalid masking, unchanged labels and valid masks, backward-compatible paths and checkpoints, and one complete/incomplete **CPU training run** with a tiny adapter. Production STLAformer/PatchSTG training and four-severity evaluations require user-side materialized caches and vendor models.
