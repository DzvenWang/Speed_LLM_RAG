# Baseline Protocol v1.1.0 — Core-5

Baseline v1.1.0 is a clean consumer of Protocol v2.1.0, ExperimentCommon v1.2.0 and the frozen Backbone v1.0.10 PatchSTG reference.

## Main sensing families

1. `entry_random`
2. `sensor_outage`
3. `temporal_outage`
4. `regional_outage`
5. `directional_outage`

The two predictive outage families are not part of the main Baseline Train/Val/Test protocol.

## Train / validation

- Protocol: `protocol_core5_s40`
- Severity: 40%
- Development train: 12,000 = 2,400/family
- Development val: 5,000 = 1,000/family
- Formal train: 50,000 = 10,000/family
- Formal val: 5,000 = 1,000/family
- Checkpoint/model selection is validation-only.

## Evaluation

After Formal selection and protocol freeze, each selected model is evaluated on the paired frozen caches:

- `protocol_core5_s20`
- `protocol_core5_s40`
- `protocol_core5_s60`
- `protocol_core5_s90`

Each severity contains 5,000 queries = 1,000/family. One evaluation PBS job per method/seed runs all four severities and writes severity-level and scenario-level summary CSVs.
