# BigST output-shape hotfix 1

This hotfix fixes the BigST screening failure:

`RuntimeError: The size of tensor a (388) must match the size of tensor b (12) at non-singleton dimension 2`.

The pinned official BigST code operates on node-first targets `[B, N, P]`, while the Shared Protocol Backbone runner uses `[B, P, N]`. The official BigST source is not modified. `BigSTAdapter` now converts the native output at the adapter boundary and validates the result against the batch target shape.

Also added:

- `adapter_revision=bigst_bnp_to_bpn_hotfix1` in BigST logs/checkpoints/summaries.
- explicit prediction/target shape assertions in training and evaluation.
- `scripts/smoke_bigst_shape_contract.py`.
- retained the earlier `setup_official_models.sh` `set -u` hotfix.

No Protocol, cache, model recipe, vendor source, or other backbone adapter is changed. Existing screening cache remains valid. Only BigST needs to be resubmitted.
