#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, os, sys, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

GADI_ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from baseline.runners.common import make_run_dir, masked_mse, require_gpu, save_val_outputs, scale_stats, set_seed, validate_stage


class ReMFTDataset(Dataset):
    def __init__(self,cache:Path,split:str):
        self.x=np.load(cache/f'{split}_x_obs_norm.npy',mmap_mode='r')
        self.y=np.load(cache/f'{split}_y_norm.npy',mmap_mode='r')
        self.valid=np.load(cache/f'{split}_future_valid.npy',mmap_mode='r')
    def __len__(self): return int(self.x.shape[0])
    def __getitem__(self,i):
        # Official ReMFT convention is [sensor, time].
        return (np.array(self.x[i].T,np.float32,copy=True),np.array(self.y[i].T,np.float32,copy=True),np.array(self.valid[i].T,np.uint8,copy=True))


@contextmanager
def chdir(path:Path):
    old=Path.cwd(); os.chdir(path)
    try: yield
    finally: os.chdir(old)


def load_official_namespace() -> dict:
    """Execute architecture cells only, stopping before official data/test code."""
    nb_path=ROOT/'vendor/ReMFT/run.ipynb'
    obj=json.loads(nb_path.read_text())
    ns={'__name__':'remft_official_arch'}
    executed=0
    with chdir(nb_path.parent):
        for cell in obj.get('cells',[]):
            if cell.get('cell_type')!='code': continue
            code=''.join(cell.get('source',[]))
            if 'data_file = os.path.join' in code or '# load data' in code:
                break
            if not code.strip(): continue
            exec(compile(code,f'{nb_path.name}:cell{executed}','exec'),ns,ns); executed+=1
    for name in ['MultiScaleSeasonMixing','MultiScaleTrendMixing','PHI','llmodel']:
        if name not in ns: raise RuntimeError(f'Official ReMFT notebook architecture missing {name}')
    return ns


def build_phi(ns:dict,cfg:dict) -> nn.Module:
    """Build the official PHI for the benchmark history length without namespace monkey-patching.

    The official notebook lets PHI(seq_len=...) control pooling/fusion layers, but
    constructs MultiScaleSeasonMixing/TrendMixing with their default seq_len=96.
    Rebinding those class names to lambdas breaks the classes' explicit
    ``super(ClassName, self)`` calls.  Construct PHI normally, then replace only
    its two mixer *instances* with the same official classes configured for H=12.
    """
    c=cfg['runners']['ReMFT']
    H=int(cfg['experiment']['history'])
    ds_window=int(c['down_sampling_window'])
    ds_layers=int(c['down_sampling_layers'])
    scale=ds_window ** ds_layers
    if H % scale != 0:
        raise ValueError(
            f'ReMFT history={H} must be divisible by '
            f'down_sampling_window**down_sampling_layers={scale}'
        )

    Season=ns['MultiScaleSeasonMixing']
    Trend=ns['MultiScaleTrendMixing']
    PHI=ns['PHI']
    phi=PHI(
        seq_len=H,
        k=ds_layers,
        c=ds_window,
        layers=int(c['phi_layers']),
    )
    phi.mixing_multi_scale_season=Season(
        seq_len=H,
        down_sampling_window=ds_window,
        down_sampling_layers=ds_layers,
    )
    phi.mixing_multi_scale_trend=Trend(
        seq_len=H,
        down_sampling_window=ds_window,
        down_sampling_layers=ds_layers,
    )
    return phi


class ReMFT12(nn.Module):
    def __init__(self,cfg:dict):
        super().__init__(); c=cfg['runners']['ReMFT']; H=int(cfg['experiment']['history']); P=int(cfg['experiment']['horizon'])
        ns=load_official_namespace()
        self.phi=build_phi(ns,cfg)
        repo=ROOT/'vendor/ReMFT'
        with chdir(repo):
            self.llmodel=ns['llmodel'](patch_len=int(c['patch_len']),stride=int(c['stride']),pred_len=P,seq_len=H)
    def forward(self,x): return self.llmodel(self.phi(x))


def enforce_checkpoint_consistent_bert_depth(model: nn.Module) -> dict:
    """Align the frozen ReMFT BERT runtime to the local pretrained checkpoint.

    The pinned ReMFT notebook may overwrite ``num_hidden_layers`` before
    ``from_pretrained``. With a standard 12-layer BERT checkpoint this creates
    extra randomly initialized encoder layers which are then frozen. Those
    layers are neither pretrained nor trainable and therefore are not an
    acceptable baseline. We keep the pretrained layers and remove only runtime
    layers that have no checkpoint counterpart.
    """
    bert_dir = ROOT / 'vendor/ReMFT/LLM/BERT'
    local_cfg = json.loads((bert_dir / 'config.json').read_text())
    llm = model.llmodel.llm_model
    runtime_cfg = getattr(llm, 'config', None)
    encoder = getattr(llm, 'encoder', None)
    if runtime_cfg is None or encoder is None or not hasattr(encoder, 'layer'):
        raise RuntimeError('ReMFT BERT runtime does not expose config/encoder.layer')

    local_depth = int(local_cfg['num_hidden_layers'])
    local_hidden = int(local_cfg['hidden_size'])
    local_heads = int(local_cfg['num_attention_heads'])
    runtime_hidden = int(getattr(runtime_cfg, 'hidden_size'))
    runtime_heads = int(getattr(runtime_cfg, 'num_attention_heads'))
    runtime_depth = len(encoder.layer)

    if runtime_hidden != local_hidden or runtime_heads != local_heads:
        raise RuntimeError(
            'ReMFT local/runtime BERT architecture mismatch beyond depth: '
            f'local hidden/heads={local_hidden}/{local_heads}, '
            f'runtime={runtime_hidden}/{runtime_heads}'
        )
    if runtime_depth < local_depth:
        raise RuntimeError(
            f'ReMFT runtime BERT has only {runtime_depth} layers but checkpoint config requires {local_depth}'
        )

    trimmed = 0
    if runtime_depth > local_depth:
        trimmed = runtime_depth - local_depth
        encoder.layer = nn.ModuleList(list(encoder.layer)[:local_depth])
        runtime_cfg.num_hidden_layers = local_depth
        print(
            'REMFT BERT DEPTH NORMALIZED '
            f'runtime_before={runtime_depth} checkpoint_depth={local_depth} trimmed_random_layers={trimmed}',
            flush=True,
        )

    return {
        'checkpoint_num_hidden_layers': local_depth,
        'runtime_layers_before_alignment': runtime_depth,
        'trimmed_random_runtime_layers': trimmed,
        'policy': 'checkpoint_consistent_pretrained_bert_depth',
    }


def build_model(cfg:dict,device:torch.device):
    repo=ROOT/'vendor/ReMFT'; bert=repo/'LLM/BERT'
    if not (bert/'config.json').exists():
        raise RuntimeError('ReMFT requires vendor/ReMFT/LLM/BERT. Run scripts/setup_remft_bert.sh with a local BERT checkpoint.')
    m=ReMFT12(cfg)
    depth_alignment = enforce_checkpoint_consistent_bert_depth(m)
    # The official notebook loads BERT with device_map='auto'. Calling .to() on
    # an Accelerate-dispatched parent can fail when any module is managed by the
    # device map. Keep the frozen BERT exactly where the official loader placed
    # it and move only the trainable benchmark-side modules to cuda:0.
    llm_device=next(m.llmodel.llm_model.parameters()).device
    if llm_device != device:
        raise RuntimeError(
            f'ReMFT official BERT was placed on {llm_device}, expected {device}. '
            'Use the single-GPU PBS job and ensure the local BERT fits on that GPU.'
        )
    m.phi.to(device)
    for name in ['mapping_layer','patch_embedding','reprogramming_layer','output_projection']:
        getattr(m.llmodel,name).to(device)
    m._llm_contract = {**llm_contract_metadata(m), **depth_alignment}
    return m



def llm_contract_metadata(model: nn.Module) -> dict:
    """Audit the pinned ReMFT LLM without changing its architecture.

    The frozen LLM must exactly match the local pretrained checkpoint after
    checkpoint-depth alignment. Any remaining architecture mismatch is a hard
    failure, as is any trainable BERT parameter.
    """
    bert_dir = ROOT / 'vendor/ReMFT/LLM/BERT'
    local_cfg = json.loads((bert_dir / 'config.json').read_text())
    llm = model.llmodel.llm_model
    runtime_cfg = getattr(llm, 'config', None)
    params = list(llm.parameters())
    frozen = bool(params) and all(not p.requires_grad for p in params)
    if not frozen:
        trainable = [name for name, p in llm.named_parameters() if p.requires_grad]
        raise RuntimeError(
            'ReMFT frozen-LLM contract failed; BERT has trainable parameters: '
            + str(trainable[:20])
        )
    runtime_layers = getattr(runtime_cfg, 'num_hidden_layers', None)
    encoder = getattr(llm, 'encoder', None)
    encoder_layers = len(encoder.layer) if encoder is not None and hasattr(encoder, 'layer') else None
    meta = {
        'local_model_path': str(bert_dir),
        'local_num_hidden_layers': local_cfg.get('num_hidden_layers'),
        'local_hidden_size': local_cfg.get('hidden_size'),
        'local_num_attention_heads': local_cfg.get('num_attention_heads'),
        'runtime_num_hidden_layers': runtime_layers,
        'runtime_encoder_layer_count': encoder_layers,
        'runtime_hidden_size': getattr(runtime_cfg, 'hidden_size', None),
        'runtime_num_attention_heads': getattr(runtime_cfg, 'num_attention_heads', None),
        'llm_frozen': frozen,
        'runtime_matches_local_depth': (
            runtime_layers == local_cfg.get('num_hidden_layers')
            if runtime_layers is not None and local_cfg.get('num_hidden_layers') is not None
            else None
        ),
        'policy': 'checkpoint_consistent_pretrained_bert_depth',
    }
    print('REMFT LLM CONTRACT ' + json.dumps(meta, default=str), flush=True)
    if meta['runtime_matches_local_depth'] is not True:
        raise RuntimeError(
            'ReMFT frozen-LLM checkpoint/runtime depth mismatch remains after alignment: ' + str(meta)
        )
    if meta['runtime_hidden_size'] != meta['local_hidden_size'] or meta['runtime_num_attention_heads'] != meta['local_num_attention_heads']:
        raise RuntimeError('ReMFT frozen-LLM hidden/head configuration does not match local checkpoint')
    return meta


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    names={name for name,p in model.named_parameters() if p.requires_grad}
    sd=model.state_dict()
    return {k:v.detach().cpu().clone() for k,v in sd.items() if k in names}


def load_trainable_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    result=model.load_state_dict(state,strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f'ReMFT checkpoint has unexpected keys: {result.unexpected_keys[:10]}')


def predict(model,dl,device,mean,std):
    model.eval(); out=[]
    with torch.no_grad():
        for x,_y,_v in dl:
            p=model(x.float().to(device)) # [B,N,P]
            if not torch.isfinite(p).all(): raise RuntimeError('ReMFT produced non-finite prediction')
            out.append((p.transpose(1,2).cpu().numpy()*np.float32(std)+np.float32(mean)).astype(np.float32))
    return np.concatenate(out,0)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['development','formal'],required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    cfg,cache=validate_stage(a.stage,'ReMFT'); set_seed(a.seed); device=require_gpu(); c=cfg['runners']['ReMFT']
    tr=ReMFTDataset(cache,'train'); va=ReMFTDataset(cache,'val')
    trl=DataLoader(tr,batch_size=int(c['batch_size']),shuffle=True,num_workers=0,pin_memory=True); val=DataLoader(va,batch_size=int(c['batch_size']),shuffle=False,num_workers=0,pin_memory=True)
    model=build_model(cfg,device); trainable=[p for p in model.parameters() if p.requires_grad]; nparams=sum(p.numel() for p in trainable)
    opt=torch.optim.Adam(trainable,lr=float(c['learning_rate']),weight_decay=float(c['weight_decay']),eps=1e-4)
    accum=max(1,int(c['gradient_accumulation_steps']))
    if a.smoke:
        x,y,v=next(iter(trl)); opt.zero_grad(set_to_none=True); p=model(x.float().to(device)); loss=masked_mse(p,y.float().to(device),v.to(device)); loss.backward(); nn.utils.clip_grad_norm_(trainable,float(c['gradient_clip'])); opt.step()
        exp=(int(c['batch_size']),tr.x.shape[2],int(cfg['experiment']['horizon']))
        if tuple(p.shape)!=exp or not torch.isfinite(p).all(): raise RuntimeError(f'ReMFT smoke shape {tuple(p.shape)} != {exp}')
        print('REMFT GPU SMOKE PASS',{'loss':float(loss),'shape':tuple(p.shape),'trainable_params':nparams},flush=True); return
    best=float('inf'); state=None; be=0; stale=0; t0=time.time()
    for ep in range(1,int(c['max_epochs'])+1):
        model.train(); opt.zero_grad(set_to_none=True); tt=tn=0.0
        for bi,(x,y,v) in enumerate(trl,1):
            p=model(x.float().to(device)); loss=masked_mse(p,y.float().to(device),v.to(device)); (loss/accum).backward(); tt+=float(loss.item()); tn+=1
            if bi%accum==0 or bi==len(trl): nn.utils.clip_grad_norm_(trainable,float(c['gradient_clip'])); opt.step(); opt.zero_grad(set_to_none=True)
        model.eval(); vt=vn=0.0
        with torch.no_grad():
            for x,y,v in val:
                p=model(x.float().to(device)); loss=masked_mse(p,y.float().to(device),v.to(device)); vt+=float(loss.item()); vn+=1
        score=vt/max(vn,1); print(f'EPOCH {ep:03d} train_MSE_norm={tt/max(tn,1):.6f} val_MSE_norm={score:.6f}',flush=True)
        if score<best-1e-8: best=score; be=ep; stale=0; state=trainable_state_dict(model)
        else:
            stale+=1
            if stale>=int(c['patience']): print(f'EARLY STOP epoch={ep} best_epoch={be}',flush=True); break
    if state is None: raise RuntimeError('ReMFT produced no checkpoint')
    load_trainable_state_dict(model,state); mean,std=scale_stats(cache); raw=predict(model,val,device,mean,std)
    rd=make_run_dir(cfg,'ReMFT',a.stage,a.seed); torch.save({'model':state,'checkpoint_format':'trainable_state_only_v1','best_epoch':be,'best_val_mse_norm':best,'history':int(cfg['experiment']['history']),'horizon':int(cfg['experiment']['horizon']),'llm_contract':getattr(model,'_llm_contract',{})},rd/'best.pt')
    sm=save_val_outputs(cfg,cache,rd,raw,'ReMFT',a.seed,a.stage,{
        'best_epoch':be,'best_val_MSE_norm':best,'trainable_params':nparams,'wall_seconds':time.time()-t0,
        'official_architecture':'vendor/ReMFT/run.ipynb architecture cells through Model definition',
        'adaptation':'official PHI + BERT reprogramming; seq_len 96->12 via post-construction official mixer instances; frozen BERT depth aligned to the local pretrained checkpoint; incomplete history fed directly; no stock data/train/test cells executed',
        'table_group':'main','checkpoint_selection':'Val masked MSE only',
        'llm_contract':getattr(model,'_llm_contract',{}),'selection_valid':True,'test_access':False,
    })
    print('REMFT COMPLETE',json.dumps({'run_dir':str(rd),'val':sm['val']},default=str))


if __name__=='__main__': main()
