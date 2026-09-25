# Migration to Protocol v2.0.1

1. Archive or remove old `Protocol_artifacts/TraffiDent_Net_06/protocol_v1` and old `protocol_eval_sXX` artifacts if they exist.
2. Replace the `Protocol` source directory with this release.
3. Keep TraffiDent-SupportBench v1.2.6 data unchanged.
4. Run:

```bash
python scripts/build_multiseverity_protocols.py --force
```

Expected active artifacts:

- `protocol_s40/development`
- `protocol_s40/formal`
- `protocol_s20/evaluation`
- `protocol_s40/evaluation`
- `protocol_s60/evaluation`
- `protocol_s90/evaluation`
- `_paired/TraffiDent_Net_06_multiseverity_v1/evaluation_pairs.csv`

Evaluation/Test manifests are frozen but access remains locked.
