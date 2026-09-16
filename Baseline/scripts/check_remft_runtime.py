#!/usr/bin/env python3
from pathlib import Path
import importlib.util, json, re, sys

GADI=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT=GADI if GADI.exists() else Path(__file__).resolve().parents[1]
repo=ROOT/'vendor/ReMFT'
nb=repo/'run.ipynb'
bert=repo/'LLM/BERT'
failed=False
for p in [nb, bert/'config.json']:
    ok=p.exists(); print(('PASS' if ok else 'FAIL'),p); failed |= not ok
weights=[bert/'model.safetensors',bert/'pytorch_model.bin']
ok=any(p.exists() for p in weights); print(('PASS' if ok else 'FAIL'),'BERT weights',*[str(p) for p in weights]); failed |= not ok
for mod in ['torch','transformers','accelerate','einops','pandas','yaml']:
    ok=importlib.util.find_spec(mod) is not None; print(('PASS' if ok else 'FAIL'),mod); failed |= not ok
if failed:
    print('Set REMFT_BERT_PATH and run scripts/setup_remft_bert.sh after cloning official models.',file=sys.stderr)
    raise SystemExit(2)

local_cfg=json.loads((bert/'config.json').read_text())
print('PASS ReMFT local BERT config',{
    'num_hidden_layers':local_cfg.get('num_hidden_layers'),
    'hidden_size':local_cfg.get('hidden_size'),
    'num_attention_heads':local_cfg.get('num_attention_heads'),
    'model_type':local_cfg.get('model_type'),
})

# Static checkpoint provenance audit when safetensors metadata is available.
safe=bert/'model.safetensors'
if safe.exists() and importlib.util.find_spec('safetensors') is not None:
    from safetensors import safe_open
    layers=set()
    with safe_open(str(safe),framework='pt',device='cpu') as f:
        for key in f.keys():
            m=re.search(r'encoder\.layer\.(\d+)\.',key)
            if m: layers.add(int(m.group(1)))
    if layers:
        print('PASS ReMFT checkpoint encoder-layer provenance',{
            'count':len(layers),'min':min(layers),'max':max(layers)
        })

# Runtime contract: the 12-step PHI adaptation must construct and execute before
# a GPU/BERT job is submitted. This catches the former namespace lambda patch
# while preserving the official ReMFT mechanism.
sys.path.insert(0,str(ROOT))
import torch
import yaml
from baseline.runners.remft_runner import load_official_namespace, build_phi

cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
ns=load_official_namespace()
phi=build_phi(ns,cfg).cpu().eval()
H=int(cfg['experiment']['history'])
with torch.no_grad():
    x=torch.randn(2,8,H,dtype=torch.float32)
    y=phi(x)
if tuple(y.shape) != tuple(x.shape):
    raise RuntimeError(f'ReMFT PHI-12 shape contract failed: input={tuple(x.shape)} output={tuple(y.shape)}')
if not torch.isfinite(y).all():
    raise RuntimeError('ReMFT PHI-12 produced non-finite output')
if int(phi.mixing_multi_scale_season.seq_len) != H:
    raise RuntimeError('ReMFT season mixer did not adopt benchmark history length')
if int(phi.mixing_multi_scale_trend.seq_len) != H:
    raise RuntimeError('ReMFT trend mixer did not adopt benchmark history length')
print('PASS ReMFT PHI-12 construction/forward',tuple(y.shape))
print('INFO full runtime LLM checkpoint-depth/frozen contract is enforced by remft_runner.py')
print('REMFT RUNTIME PASS')
