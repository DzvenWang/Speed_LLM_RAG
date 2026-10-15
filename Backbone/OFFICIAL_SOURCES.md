# Official source verification

The legacy `vendor/SOURCE_LOCK.json` pins the official PatchSTG repository:

- repository: `https://github.com/LMissher/PatchSTG.git`
- commit: `8b298b230227a304a082e05241c69dca3740a73d`

The source-tree SHA256 excludes `.git` and Python cache files. The preflight
checks the **selected model** only. By default a tree-hash mismatch prints
`WARNING SOURCE_TREE_HASH_MISMATCH` and continues, without changing the lock
file. Missing selected vendor, malformed lock JSON, and a PatchSTG
commit/remote identity conflict remain fatal.

```bash
python Backbone/scripts/source_lock.py verify --model STLAformer
python Backbone/scripts/source_lock.py verify --model PatchSTG
```

Currently only PatchSTG is recorded in the legacy source-lock file. For other
models, verification checks that the vendor directory exists and reports
`WARNING SOURCE_LOCK_ENTRY_MISSING` instead of checking a non-existent hash.

To require an exact tree match (e.g. prior to preserving a stripped vendor):

```bash
python Backbone/scripts/source_lock.py verify --model PatchSTG --strict-hash
```

A mismatch in relaxed mode is **not proof of source equivalence**. Preserve
original lock metadata for reproducible historical experiments.
