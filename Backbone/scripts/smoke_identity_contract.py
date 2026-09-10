#!/usr/bin/env python3
from pathlib import Path
import json, tempfile, sys, yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from backbone.identity import load_cache_identity, require_identity, require_cache_version_compatibility
cfg=yaml.safe_load((ROOT/'configs/backbone_selection.yaml').read_text())
with tempfile.TemporaryDirectory() as td:
    p=Path(td); m={'project_version':'0.3.1','backbone_stage':'screening','protocol_stage':'development','protocol_stage_hash':'P','cache_fingerprint':'C','split_view_hashes':{'train':'T','val':'V'},'subgraph':'TraffiDent_Net_06'}
    (p/'cache_manifest.json').write_text(json.dumps(m)); ident=load_cache_identity(p)
    require_cache_version_compatibility(cfg,ident,'screening')
    require_identity({'run_identity':ident},ident)
    try: require_identity({'project_version':'0.3.0'},ident)
    except RuntimeError: pass
    else: raise AssertionError('Old/untracked result was not rejected')
    try: require_cache_version_compatibility(cfg,ident,'formal')
    except RuntimeError: pass
    else: raise AssertionError('v0.3.1 cache was incorrectly accepted for Formal')
print('BACKBONE IDENTITY CONTRACT SMOKE PASS')
