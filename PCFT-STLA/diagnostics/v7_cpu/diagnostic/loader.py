"""Strict Gadi checkpoint and canonical-cache loading; never intentionally rebuild caches."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import torch
from pcft_stla.v6.config import root,sha256,make_paths
from pcft_stla.v6.data import open_bench,MultiTargetSplit
from pcft_stla.v6.graph import source_identity,graph_root
from pcft_stla.v6.pipeline import prepare
from pcft_stla.v7.pipeline import instantiate


def assert_existing_inputs(cfg):
    """Fail BEFORE v6.prepare can materialize train/evaluation/graph caches."""
    target=cfg['experiment']['target'];dataset=cfg['experiment']['dataset']
    repo,formal,eval_paths=make_paths(cfg,target,dataset)
    bench=open_bench(repo,dataset,target=target)
    all_dirs={'train':formal,'val':formal}
    all_dirs.update({f'test_s{s:02d}':path for s,path in eval_paths.items()})
    store=repo/'PCFT-STLA_cache'/'v6'/dataset/target
    multi_roots={}
    for label,path in all_dirs.items():
        split='test' if label.startswith('test_') else label
        mroot=store/(label+'_'+sha256(path/'cache_manifest.json')[:12])
        required=[mroot/'identity.json',mroot/'x.npy',mroot/'obs.npy',mroot/'age.npy',path/f'{split}_manifest.csv']
        missing=[str(p) for p in required if not p.is_file()]
        if missing:raise FileNotFoundError('CPU diagnostic is read-only; missing existing multichannel cache: '+str(missing))
        multi_roots[label]=mroot
    train=MultiTargetSplit(formal,'train',multi_roots['train'])
    identity=source_identity(bench,train,dict(graph_config=cfg['graph'],multi_cache_root=str(multi_roots['train'])))
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    base=graph_root(repo,dataset,target)
    graph_file=base/(key[:16]+'.npz');meta_file=base/(key[:16]+'.json')
    if not graph_file.is_file() or not meta_file.is_file():
        raise FileNotFoundError('CPU diagnostic requires prebuilt matching graph, not regeneration: '+str(graph_file))
    info=json.loads(meta_file.read_text())
    if info.get('source_identity')!=identity or info.get('sha256')!=sha256(graph_file):
        raise RuntimeError('Graph cache identity mismatch; refusing to rebuild in diagnostic mode')
    return {'cache_roots':multi_roots,'graph_file':str(graph_file)}


def load_model(checkpoint,backbone,device=torch.device('cpu')):
    if device.type!='cpu':raise ValueError('CPU diagnostic refuses non-CPU devices')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    checkpoint=Path(checkpoint).resolve()
    backbone=Path(backbone).resolve()
    if not checkpoint.is_file() or not backbone.is_file():raise FileNotFoundError((checkpoint,backbone))
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if cp.get('schema')!='pcft_v7_best_v1':raise ValueError('Checkpoint is not a v7 best-stage checkpoint')
    cfg=cp['config']
    target=cfg['experiment']['target'];os.environ['TRAFFIDENT_TARGET']=target
    if cfg['experiment']['train_mask_mode']!='fixed_s40':
        # A trained mixed-mask model can be evaluated on the existing val/test caches, but the identity must match.
        print('WARN_TRAINED_WITH_MIXED_MASKS',flush=True)
    checked=assert_existing_inputs(cfg)
    repo,_,actual_backbone,graph,ds,identity=prepare(cfg,str(backbone),smoke=False)
    identity=dict(identity,v7_schema='pcft_v7_anchor_selective_v1',model_config=cfg['model'],train_mask_mode=cfg['experiment']['train_mask_mode'])
    if cp.get('identity')!=identity:raise RuntimeError('v7 checkpoint identity, graph or Backbone hash mismatch')
    model=instantiate(cfg,repo,actual_backbone,graph,identity,device)
    model.load_state_dict(cp['model'],strict=True)
    model.anchor.set_trainable_tail(0)
    model.set_stage('joint');model.eval()
    return model,ds,identity,cfg,graph,checked
