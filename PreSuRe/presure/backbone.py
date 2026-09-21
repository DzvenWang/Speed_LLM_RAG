from __future__ import annotations
from pathlib import Path
import json, sys
import numpy as np, torch
from torch.utils.data import DataLoader
from .contract import cache_dir, load_cache_identity, compare_checkpoint_identity
from .utils import sha256_file, log

def _resolve_dev_checkpoint(cfg):
    root=Path(cfg['project']['backbone_exp_root'])/'PatchSTG'/cfg['experiment']['subgraph']/'screening'
    target=load_cache_identity(cfg,'development'); cand=[]; rejected=[]
    for s in root.glob('*/summary.json') if root.exists() else []:
        try:
            j=json.loads(s.read_text()); ck=s.parent/'best.pt'
            if int(j.get('seed',-1))!=int(cfg['experiment']['reference_seed']) or not ck.exists(): continue
            if j.get('model')!='PatchSTG' or j.get('stage')!='screening': continue
            obj=torch.load(ck,map_location='cpu',weights_only=False); compare_checkpoint_identity(obj,target,'development')
            cand.append((s.stat().st_mtime,ck))
        except Exception as e: rejected.append((str(s),str(e)))
    if not cand:
        msg='No completed current-contract PatchSTG screening seed10 checkpoint found for Development.'
        if rejected: msg+=f' Rejected {len(rejected)} stale/mismatched run(s).'
        raise RuntimeError(msg)
    return max(cand,key=lambda x:x[0])[1], 'screening_seed10_current_contract'

def resolve_checkpoint(cfg,stage,override=None):
    if override:
        p=Path(override); source='explicit_override'
    elif stage=='development':
        p,source=_resolve_dev_checkpoint(cfg)
    elif stage=='formal':
        f=Path(cfg['project']['backbone_exp_root'])/'FROZEN_BACKBONE.json'
        if not f.exists(): raise RuntimeError('Formal PreSuRe requires Backbone_exps/FROZEN_BACKBONE.json')
        j=json.loads(f.read_text())
        if j.get('selected_model')!='PatchSTG' or int(j.get('reference_seed',-1))!=int(cfg['experiment']['reference_seed']): raise RuntimeError('Frozen backbone is not configured PatchSTG reference seed10')
        p=Path(j['reference_checkpoint']); source='FROZEN_BACKBONE.json'
        if sha256_file(p)!=str(j.get('reference_checkpoint_sha256','')): raise RuntimeError('Frozen checkpoint SHA256 mismatch')
    else: raise ValueError(stage)
    if not p.exists(): raise FileNotFoundError(p)
    ck=torch.load(p,map_location='cpu',weights_only=False)
    if ck.get('model')!='PatchSTG': raise RuntimeError(f'Expected PatchSTG checkpoint, got {ck.get("model")}')
    if int(ck.get('seed',-1))!=int(cfg['experiment']['reference_seed']): raise RuntimeError('PreSuRe requires fixed PatchSTG seed10 reference checkpoint')
    ident=load_cache_identity(cfg,stage); compare_checkpoint_identity(ck,ident,stage)
    return p,ck,source

class CachedTorch(torch.utils.data.Dataset):
    def __init__(self,cache,split):
        self.r=Path(cache); self.split=split
        self.x=np.load(self.r/f'{split}_x_obs_norm.npy',mmap_mode='r')
        self.y=np.load(self.r/f'{split}_y_raw.npy',mmap_mode='r')
        self.valid=np.load(self.r/f'{split}_future_valid.npy',mmap_mode='r')
        self.direct=np.load(self.r/f'{split}_direct_mask.npy',mmap_mode='r')
        self.support=np.load(self.r/f'{split}_support_mask.npy',mmap_mode='r')
        self.tod=np.load(self.r/f'{split}_tod_hist.npy',mmap_mode='r'); self.dow=np.load(self.r/f'{split}_dow_hist.npy',mmap_mode='r')
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        return {'x':torch.from_numpy(np.asarray(self.x[i]).copy()),'y_raw':torch.from_numpy(np.asarray(self.y[i]).copy()),
            'future_valid':torch.from_numpy(np.asarray(self.valid[i],bool).copy()),'direct_mask':torch.from_numpy(np.asarray(self.direct[i],bool).copy()),
            'support_mask':torch.from_numpy(np.asarray(self.support[i],bool).copy()),'tod_hist':torch.from_numpy(np.asarray(self.tod[i],np.int64).copy()),
            'dow_hist':torch.from_numpy(np.asarray(self.dow[i],np.int64).copy())}

def load_patchstg_adapter(cfg,stage,device,override=None):
    broot=Path(cfg['project']['backbone_root']); sroot=Path(cfg['project']['supportbench_root'])
    for p in [broot,sroot]:
        if str(p) not in sys.path: sys.path.insert(0,str(p))
    from backbone.adapters import build_adapter
    from supportbench.dataset import SupportBench
    path,ck,source=resolve_checkpoint(cfg,stage,override)
    b=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root'])
    adapter=build_adapter('PatchSTG',broot/'vendor',b.N,device,float(ck['mean']),float(ck['std']),ck['model_config'],b.adj,b.nodes)
    adapter.model.load_state_dict(ck['state_dict']); adapter.model.eval()
    log(f'PATCHSTG checkpoint={path} source={source} seed={ck["seed"]} best_epoch={ck.get("best_epoch")}')
    return adapter,path,ck,source

@torch.no_grad()
def infer_split(adapter,cache,split,batch_size,device,runtime=None):
    ds=CachedTorch(cache,split); runtime=runtime or {}; nw=int(runtime.get('num_workers',0)); pin=bool(runtime.get('pin_memory',True)) and torch.cuda.is_available()
    kw={'num_workers':nw,'pin_memory':pin}
    if nw>0: kw.update(prefetch_factor=int(runtime.get('prefetch_factor',2)), persistent_workers=bool(runtime.get('persistent_workers',True)))
    dl=DataLoader(ds,batch_size=batch_size,shuffle=False,drop_last=False,**kw)
    out=[]
    for bi,b in enumerate(dl,1):
        pn=adapter.eval_forward(b); pr=adapter.raw(pn).detach().cpu().numpy().astype(np.float32); out.append(pr)
        if bi%100==0: log(f'PatchSTG inference {split} batches={bi}/{len(dl)}')
    return np.concatenate(out,axis=0)



def _hook_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x,(list,tuple)):
        for z in x:
            t=_hook_tensor(z)
            if t is not None: return t
    if isinstance(x,dict):
        for z in x.values():
            t=_hook_tensor(z)
            if t is not None: return t
    return None


def _node_align_hidden(h, n_nodes):
    t=_hook_tensor(h)
    if t is None: raise RuntimeError('Hidden hook returned no tensor')
    if t.ndim==4 and t.shape[-1]==1:
        t=t.squeeze(-1)
    if t.ndim==2:
        raise RuntimeError(f'Hidden output {tuple(t.shape)} has no node axis')
    if t.ndim==3:
        if t.shape[1]==n_nodes: return t
        if t.shape[2]==n_nodes: return t.permute(0,2,1).contiguous()
    # Conservative fallback for [B,C,N,T] or [B,N,C,T]: average only auxiliary axes.
    if t.ndim>=4:
        dims=list(t.shape)
        candidates=[i for i in range(1,t.ndim) if dims[i]==n_nodes]
        if len(candidates)==1:
            ni=candidates[0]
            perm=[0,ni]+[i for i in range(1,t.ndim) if i!=ni]
            u=t.permute(*perm).contiguous()
            return u.flatten(2).mean(-1,keepdim=True) if u.flatten(2).shape[-1]==1 else u.flatten(2)
    raise RuntimeError(f'Cannot node-align hidden output shape {tuple(t.shape)} with N={n_nodes}')


@torch.no_grad()
def infer_split_with_explicit_hidden(adapter, cache, split, batch_size, device, module_name='input_st_fc', runtime=None):
    """Run PatchSTG and capture one explicitly named module only.

    This is intentionally not an automatic layer sweep. If the requested module does not
    exist or cannot be node-aligned, the caller can mark the hidden view unavailable.
    """
    ds=CachedTorch(cache,split); runtime=runtime or {}; nw=int(runtime.get('num_workers',0)); pin=bool(runtime.get('pin_memory',True)) and torch.cuda.is_available()
    kw={'num_workers':nw,'pin_memory':pin}
    if nw>0: kw.update(prefetch_factor=int(runtime.get('prefetch_factor',2)), persistent_workers=bool(runtime.get('persistent_workers',True)))
    dl=DataLoader(ds,batch_size=batch_size,shuffle=False,drop_last=False,**kw)
    modules=dict(adapter.model.named_modules())
    if module_name not in modules:
        return infer_split(adapter,cache,split,batch_size,device,runtime), None, f'module_not_found:{module_name}'
    captured=[]
    def hook(_m,_i,o): captured.append(o)
    handle=modules[module_name].register_forward_hook(hook)
    preds=[]; hiddens=[]; reason='ok'
    try:
        for bi,b in enumerate(dl,1):
            captured.clear()
            pn=adapter.eval_forward(b)
            pr=adapter.raw(pn).detach().cpu().numpy().astype(np.float32)
            preds.append(pr)
            if not captured:
                raise RuntimeError(f'Explicit hidden module {module_name} was not executed')
            h=_node_align_hidden(captured[-1], pr.shape[-1]).detach().cpu().numpy().astype(np.float32)
            if h.ndim!=3 or h.shape[1]!=pr.shape[-1]:
                raise RuntimeError(f'Node-aligned hidden has unexpected shape {h.shape}')
            hiddens.append(h)
            if bi%100==0: log(f'PatchSTG inference+hidden {split} batches={bi}/{len(dl)} hidden={h.shape}')
    except Exception as e:
        reason=f'hidden_capture_failed:{type(e).__name__}:{e}'
        log(f'WARN explicit hidden unavailable: {reason}')
        # Predictions collected before a hook failure are discarded and recomputed cleanly.
        handle.remove()
        return infer_split(adapter,cache,split,batch_size,device,runtime), None, reason
    finally:
        try: handle.remove()
        except Exception: pass
    return np.concatenate(preds,axis=0), np.concatenate(hiddens,axis=0), reason
