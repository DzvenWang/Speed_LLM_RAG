from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / 'baseline/backbones/patchstg_bridge.py').read_text()
assert "'verify-one'" not in text, 'PatchSTG bridge must not call Backbone with Baseline-only verify-one API'
assert "'verify'" in text and "'--config'" in text, 'PatchSTG bridge must use Backbone native source_lock verify --config API'
assert "configs/backbone.yaml" in text, 'PatchSTG bridge must validate/load the current Backbone config'
assert "configs/backbone_selection.yaml" not in text, 'PatchSTG bridge must not use stale backbone_selection.yaml'
print('SMOKE PatchSTG bridge Backbone source-lock API PASS')
