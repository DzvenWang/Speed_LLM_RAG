# Baseline v1.1.0 — Core-5 migration

Main benchmark protocol:

- Train/Val: `protocol_core5_s40`, severity 40%, five Core families.
- Evaluate: paired frozen `protocol_core5_s20/s40/s60/s90`.
- Families: `entry_random`, `sensor_outage`, `temporal_outage`, `regional_outage`, `directional_outage`.
- Development: 12,000 train (2,400/family), 5,000 val (1,000/family).
- Formal: 50,000 train (10,000/family), 5,000 val (1,000/family).
- Evaluation: 5,000 test per severity (1,000/family), 20,000 test queries across four severities.

Old v1.0.0 formal runs are intentionally not selectable because Baseline project version and shared cache identity changed.
The current frozen PatchSTG must come from Backbone v1.0.10 Core-5 formal s40 and ExperimentCommon v1.2.0 / Protocol v2.1.0.

Evaluation is submitted once per method/seed. One PBS job evaluates s20 -> s40 -> s60 -> s90 and writes:

- `Baseline_exps/evaluation/<method>/TraffiDent_Net_06/seed<seed>/severity_summary.csv`
- `Baseline_exps/evaluation/<method>/TraffiDent_Net_06/seed<seed>/scenario_summary.csv`

Each severity retains its own prediction and metric files under `.../seed<seed>/sXX/`.
