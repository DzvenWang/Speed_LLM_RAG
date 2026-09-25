# Protocol v2.0.2

- Removed the `EVALUATION_ACCESS_GRANTED.json` access gate from `shared_protocol.consumer`.
- Frozen/hash/view-integrity checks remain unchanged.
- Multi-severity pairing remains unchanged.
- `grant_evaluation_access.py`, `revoke_evaluation_access.py`, and `authorize_evaluation.py` are now compatibility no-ops.
- Future configs use `evaluation_access_policy: frozen_manifest_direct_access`.
- Existing v2.0.1 frozen artifacts remain consumable; their hashes/manifests do not need to be rebuilt solely for this lock removal.
