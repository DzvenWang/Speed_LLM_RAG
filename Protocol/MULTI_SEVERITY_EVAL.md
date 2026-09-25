# Multi-Severity Evaluation (Protocol v2.1.0)

The active protocol family is:

- `protocol_core5_s20` — paired Evaluation only
- `protocol_core5_s40` — Development + Formal + paired Evaluation
- `protocol_core5_s60` — paired Evaluation only
- `protocol_core5_s90` — paired Evaluation only

All four Test manifests share one frozen pairing manifest under `_paired/TraffiDent_Net_06_core5_multiseverity_v1/`.

Build/freeze all manifests without granting Test access:

```bash
python scripts/build_multiseverity_protocols.py --force
```

Final Test access is a separate action:

```bash
python scripts/grant_evaluation_access.py --acknowledge-models-frozen
```
