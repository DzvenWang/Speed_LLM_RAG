"""ReShift stage s0: train the STLA mask projection (S0) from scratch.

Ported from the CoRSTA v0.1 bootstrap (corsta/v7/runner.py::s0_step) so that
ReShift does not consume any CoRSTA output: frozen official STLAformer, only
the 4->d->d mask projection (zero-initialised last layer) is optimised with a
masked MAE on normalised targets; selection = raw MAE on the chronological
first 60% of Validation windows.  Benchmark (as-shipped) masks are used, as in
the original bootstrap.
"""
from __future__ import annotations
import copy,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from corsta.data import SupportSplit
from corsta.stla import load_pretrained_stla
from corsta.student import SplitSTLA
from reshift.util import atomic_json,atomic_torch,checkpoint_load,rng_state,restore_rng,sha256,to_device


def _selection(ds):
    starts=np.asarray(ds.window_starts[:len(ds)],dtype=np.int64)
    unique=np.unique(starts)
    if len(unique)<3:raise RuntimeError('S0 selection needs >=3 validation windows')
    cut=unique[min(len(unique)-1,max(0,int(len(unique)*.60)-1))]
    return np.flatnonzero(starts<=cut).tolist()


def _batch(raw):
    y=torch.nan_to_num(raw['y_norm'].float())
    valid=raw['future_valid'].bool()&torch.isfinite(raw['y_norm'])
    return y,valid


def train_s0(root,backbone_checkpoint,cache,stage_dir,stage_key,cfg_s0,tail_layers,device,log,limits=None):
    stage_dir=Path(stage_dir);stage_dir.mkdir(parents=True,exist_ok=True)
    done=stage_dir/'COMPLETE.json';best_path=stage_dir/'best.pt';last_path=stage_dir/'last.pt'
    if done.exists():
        rep=json.loads(done.read_text())
        if rep.get('stage_key')!=stage_key or sha256(best_path)!=rep.get('best_sha256'):
            raise RuntimeError(f'S0 stage identity/sha mismatch in {stage_dir}')
        log(f'RESHIFT_S0_REUSE {best_path}');return best_path
    limits=limits or {}
    base,bc=load_pretrained_stla(Path(root),Path(backbone_checkpoint),device)
    stla=SplitSTLA(base,tail_layers=int(tail_layers)).to(device)
    for p in stla.parameters():p.requires_grad_(False)
    for p in stla.mask_proj.parameters():p.requires_grad_(True)
    train=SupportSplit(cache,'train',limits.get('train'))
    val=SupportSplit(cache,'val',limits.get('val'))
    if train.x_obs.shape[2]!=int(base.num_nodes):raise RuntimeError('S0 data/backbone node mismatch')
    sel=_selection(val)
    bs=int(cfg_s0.get('batch_size',64));ebs=int(cfg_s0.get('eval_batch_size',64))
    workers=int(cfg_s0.get('workers',2))
    tl=DataLoader(train,batch_size=bs,shuffle=True,num_workers=workers,pin_memory=torch.device(device).type=='cuda')
    vl=DataLoader(Subset(val,sel),batch_size=ebs,shuffle=False,num_workers=0)
    opt=torch.optim.AdamW(stla.mask_proj.parameters(),lr=float(cfg_s0.get('lr',3e-4)))
    amp=torch.device(device).type=='cuda' and bool(cfg_s0.get('amp',True))
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    mean=float(bc['mean']);std=float(bc['std']);epochs=int(cfg_s0.get('epochs',8))
    bsha=sha256(backbone_checkpoint)

    def evaluate():
        stla.eval();s=0.;c=0.
        with torch.no_grad():
            for raw in vl:
                b=to_device(raw,device)
                pred,_=stla(b['x'],b['mask'],b['tod'],b['dow'])
                v=b['future_valid'].float()*torch.isfinite(b['y_raw']).float()
                y=torch.nan_to_num(b['y_raw'].float())
                s+=float((((pred.float()*std+mean)-y).abs()*v).sum());c+=float(v.sum())
        return s/max(c,1.)

    if last_path.exists():
        ck=checkpoint_load(last_path)
        if ck.get('stage_key')!=stage_key:raise RuntimeError('S0 resume identity mismatch')
        stla.mask_proj.load_state_dict(ck['current']);opt.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler']);restore_rng(ck['rng'])
        best=float(ck['best']);best_state=ck['best_state'];best_epoch=int(ck['best_epoch']);start=int(ck['epoch'])+1
        log(f'RESHIFT_S0_RESUME epoch={start} best={best:.6f}')
    else:
        best=evaluate();best_state=copy.deepcopy(stla.mask_proj.state_dict());best_epoch=0;start=1
        log(f'RESHIFT_S0_BASELINE selection_mae={best:.6f}')
    for ep in range(start,epochs+1):
        stla.train();stla.base.eval();s=0.;c=0.
        for raw in tl:
            b=to_device(raw,device);y,valid=_batch(b)
            with torch.autocast(device_type=torch.device(device).type,dtype=torch.float16,enabled=amp):
                pred,_=stla(b['x'],b['mask'],b['tod'],b['dow'])
            w=valid.float();loss=((pred.float()-y).abs()*w).sum()/w.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward();scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(stla.mask_proj.parameters(),5.0);scaler.step(opt);scaler.update()
            else:
                loss.backward();nn.utils.clip_grad_norm_(stla.mask_proj.parameters(),5.0);opt.step()
            s+=float(loss.detach())*float(w.sum());c+=float(w.sum())
        score=evaluate()
        if score<best:best=score;best_epoch=ep;best_state=copy.deepcopy(stla.mask_proj.state_dict())
        log(f'RESHIFT_S0_EPOCH epoch={ep}/{epochs} train_norm_mae={s/max(c,1.):.6f} '
            f'selection_mae={score:.6f} best={best:.6f}')
        atomic_torch({'stage_key':stage_key,'epoch':ep,'current':stla.mask_proj.state_dict(),
                      'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'best':best,
                      'best_state':best_state,'best_epoch':best_epoch,'rng':rng_state()},last_path)
    atomic_torch({'version':'reshift-s0','mask_proj':best_state,'backbone_sha256':bsha,
                  'best_selection_mae':best,'epoch':best_epoch,'stage_key':stage_key},best_path)
    atomic_json({'stage_key':stage_key,'best_epoch':best_epoch,'best_selection_mae':best,
                 'best_sha256':sha256(best_path)},done)
    log(f'RESHIFT_S0_PASS best_epoch={best_epoch} selection_mae={best:.6f} path={best_path}')
    return best_path
