# Official source lock

The active numerical backbone source is the official PatchSTG repository:

- repository: `https://github.com/LMissher/PatchSTG.git`
- commit: `8b298b230227a304a082e05241c69dca3740a73d`

`vendor/SOURCE_LOCK.json` additionally locks the vendored source tree SHA256. Run:

```bash
python scripts/source_lock.py verify
```

before training.
