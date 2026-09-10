from pathlib import Path
import json, tempfile, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from baseline.run_identity import load_cache_identity, build_run_manifest, identity_key
with tempfile.TemporaryDirectory() as td:
    p=Path(td); m={'project_version':'0.4.0','stage':'development','protocol_stage_hash':'P','cache_fingerprint':'C','split_protocol_view_hashes':{'train':'T','val':'V'},'subgraph':'TraffiDent_Net_06'}
    (p/'cache_manifest.json').write_text(json.dumps(m)); ident=load_cache_identity(p,'0.4.0'); run=build_run_manifest(p,'0.4.0','RAFT',10,'generic_retrieval')
    assert identity_key(ident)==identity_key(run)
print('BASELINE IDENTITY CONTRACT SMOKE PASS')
