# Official numerical backbone sources — v0.5.0

The active benchmark uses the pinned source commits in `configs/backbone_selection.yaml` and `vendor/SOURCE_LOCK.json`:

- STAEformer — `fc49d39b2f1a8e3cf37b6289d7240680e1690f3f`
- HimNet — `2a9aa44d1864f6fb60bdf9d14bafd6c1681e81a5`
- BigST — `830cb529354c0cda75c1660d1a0af2ffc7e66abb`
- STDN — `e24a60fa9485fb9805df9c171d67ab60840d757d`
- PatchSTG — `8b298b230227a304a082e05241c69dca3740a73d`

The release does not modify those model architectures. TraffiDent adaptation occurs at the adapter/data boundary and all models consume the same ExperimentCommon temporal/data contract.
