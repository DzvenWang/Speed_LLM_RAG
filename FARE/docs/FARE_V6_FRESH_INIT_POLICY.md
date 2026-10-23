# FARE v6.0.0 — Strict Fresh Initialization

Policy from the research program: **reuse ONLY the paired chronological OOF predictions / Memory and the separate pretrained frozen STLAformer backbone; never initialize any trainable FARE v6 module from old FARE v5 best.pt or last.pt.**

The STLAformer backbone is a frozen pretrained reference, NOT old FARE initialization. OOF fold checkpoints are not used to initialize FARE: the paired OOF outputs provide historical training evidence.

- Both `submit_gadi.py` and `run_pipeline.py` reject the `--init-from` CLI option.
- `read_config` refuses `training.initialize_from`.
- PBS aborts if `FARE_INIT_FROM` exists.
- Fresh training refuses a run directory containing existing FARE checkpoints.
- `manifest.json` records `fare=fresh_random_seeded`, and logs emit `FARE6_INITIALIZATION`.
- `--resume` remains possible **only** as an explicit interruption recovery using a compatible FARE v6 checkpoint, never v5 weights. Fresh runs must omit it.

## Gadi Commands

```bash
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/FARE:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
python -m pytest -q FARE/tests

python -u FARE/scripts/submit_gadi.py --target speed \
  --config FARE/configs/formal.yaml \
  --checkpoint "$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt" \
  --oos-forecast-file "$PWD/FARE_cache/oof_speed_paired.npz" --check-only

python -u FARE/scripts/submit_gadi.py --target flow \
  --config FARE/configs/formal.yaml \
  --checkpoint "$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt" \
  --oos-forecast-file "$PWD/FARE_cache/oof_flow_paired.npz" --check-only

# Fresh independent jobs (NO --init-from, NO --resume):
python -u FARE/scripts/submit_gadi.py --target speed \
  --config FARE/configs/formal.yaml \
  --checkpoint "$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt" \
  --oos-forecast-file "$PWD/FARE_cache/oof_speed_paired.npz"

python -u FARE/scripts/submit_gadi.py --target flow \
  --config FARE/configs/formal.yaml \
  --checkpoint "$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt" \
  --oos-forecast-file "$PWD/FARE_cache/oof_flow_paired.npz"
```

Perform actual V100 `--smoke` before production if the new model/GPU-cache has not previously been tested.
