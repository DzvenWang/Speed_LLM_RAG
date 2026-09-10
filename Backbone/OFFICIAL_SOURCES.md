# Official numerical backbone sources — v0.4.0

All model repositories are checked out at exact commits and protected by `vendor/SOURCE_LOCK.json`.
Vendor model source is not edited; TraffiDent compatibility lives in `backbone/adapters.py`.

| Model | Venue/year | Repository | Frozen commit | Adapter note |
|---|---|---|---|---|
| STAEformer | CIKM 2023 | XDZhelheim/STAEformer | `fc49d39...` | existing |
| HimNet | KDD 2024 | XDZhelheim/HimNet | `2a9aa44...` | existing |
| BigST | 2024 | usail-hkust/BigST | `830cb529...` | `[B,N,P] -> [B,P,N]` hotfix at adapter only |
| STDN | existing | roarer008/STDN | `e24a60f...` | unused PyG import isolated |
| PatchSTG | KDD 2025 | LMissher/PatchSTG | `8b298b230227a304a082e05241c69dca3740a73d` | official KDTree/reorder functions; TraffiDent static node metadata and adjacency |
| DST-SGNN | IJCAI 2025 | komorebi424/DST-SGNN | `ce426c7f0382aecb1a09c731fdc5ad411b36dfdc` | pure numerical 2025 snapshot, before later repository LLM extensions; H=P=12 adapter |

## PatchSTG adaptation boundary

The official SD configuration already uses 12-step input and 12-step output. For Net-06 (N=388), v0.4.0 scales the official spatial partition geometry to `recur=8`, `spn=256`, `sps=2`, `factors=16`, while retaining official `kdTree()` and `reorderData()` implementation. Time-of-day slots are 288 because TraffiDent is 5-minute data.

## DST-SGNN adaptation boundary

The pinned source's experiments use longer histories, but the model supports `seq_length=12, patch_len=12`: this produces one official input patch; SGSC's unchanged `fc12` then projects that patch representation to its fixed 12 internal slots, and the official prediction head outputs P=12. The vendor code's `cuda:0` requirement is preserved, so this adapter is GPU-only.
