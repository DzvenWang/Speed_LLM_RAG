# Oracle handoff

Oracle-specific prediction export is intentionally not owned by Backbone.

`Oracle_Diagnostic/scripts/export_backbone_predictions.py` imports the frozen
Backbone model/adapters and writes diagnostic-only prediction contracts under
`Oracle_Diagnostic_exps/`.

Backbone itself remains unchanged numerically and continues to own:

- PatchSTG training and standard evaluation;
- `FROZEN_BACKBONE.json` and frozen-checkpoint integrity;
- official-source/source-lock validation;
- generic model/data adapters used by downstream consumers.
