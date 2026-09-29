#!/usr/bin/env python3
from pathlib import Path
import importlib.util, json, re, sys

GADI=next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
ROOT=GADI if GADI.exists() else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from baseline.remft_runtime import bind_remft_bert_env

repo=ROOT/'vendor/ReMFT'
nb=repo/'run.ipynb'
failed=False
ok=nb.exists(); print(('PASS' if ok else 'FAIL'),nb); failed |= not ok
try:
    bert=bind_remft_bert_env(ROOT)
    print('PASS ReMFT BERT path', bert)
except FileNotFoundError as exc:
    print('FAIL ReMFT BERT path', exc, file=sys.stderr)
    raise SystemExit(2)
for p in [bert/'config.json']:
    ok=p.exists(); print(('PASS' if ok else 'FAIL'),p); failed |= not ok
weights=[bert/'model.safetensors',bert/'pytorch_model.bin']
ok=any(p.exists() for p in weights); print(('PASS' if ok else 'FAIL'),'BERT weights',*[str(p) for p in weights]); failed |= not ok
for mod in ['torch','transformers','accelerate','einops','pandas','yaml']:
    ok=importlib.util.find_spec(mod) is not None; print(('PASS' if ok else 'FAIL'),mod); failed |= not ok
if failed:
    raise SystemExit(2)

local_cfg=json.loads((bert/'config.json').read_text())
print('PASS ReMFT local BERT config',{
    'num_hidden_layers':local_cfg.get('num_hidden_layers'),
    'hidden_size':local_cfg.get('hidden_size'),
    'num_attention_heads':local_cfg.get('num_attention_heads'),
    'model_type':local_cfg.get('model_type'),
})

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
