# Backbone v0.3.1 warning hotfix (pynvml intentionally untouched)

This is a warning-hygiene patch only. It does **not** change Shared Protocol views,
cache fingerprints, model recipes, vendor source trees, or the numerical STDN path.

Changes:
- isolate STDN's unused legacy `torch_geometric` imports so Gadi does not load an incompatible `torch-scatter` binary;
- fail loudly if that unused legacy PyG helper is unexpectedly reached;
- suppress NumPy's expected divide-by-zero notice while official STDN code immediately maps isolated-node `inf` factors to zero;
- suppress only STDN's exact tensor copy-construction warning at the adapter boundary;
- remove `torch_geometric` from the active Backbone preflight requirement because the selected STDN `first_model` path does not use it;
- unset deprecated `TRANSFORMERS_CACHE` in PBS jobs.

`pynvml` is intentionally **not** modified or filtered, per project request.
