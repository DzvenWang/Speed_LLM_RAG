# TCRR-STLA v5.0.0 release notes

- Unified the *active* model API, Python modules, scripts, PBS, YAML configuration, checkpoint version and cache/output directories under **v5**.
- Preserved all three independently controllable C1/C2/C3 mechanisms (eight Mxyz modes).
- Kept C1/C2 + C3 stage training and automatic validation diagnostics in the formal task.
- Forked S0 adapter training and Frozen Base Cache into native v5 modules; no v4 trainer/cache import required by the v5 entrypoints.
- Fresh retrieval, teacher, CF replay and shared-base evidence only; no legacy output compatibility.
- Default test outputs are CSV and JSON; NPZ predictions are opt-in.
- Older Python source modules remain as internal helpers and regression references, not as active experiments.
