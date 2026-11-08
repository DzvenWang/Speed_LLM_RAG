"""TCRR v5 self-contained S0 adapter initialization and training."""
from __future__ import annotations
import copy,json,random,time,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset
from .calibration import chronological_partitions
from ..config import file_sha256,cache_path
from ..stla import load_pretrained_stla,MaskAwareSTLAformer
from ..data import SupportSplit,RetrievalCache
from ..evidence import FutureMemory
from ..replay_cache import prepare_bank
from ..normal_teacher import prepare_normal_teacher_bank
from ..trainer import seed_all,_make_loader,_batch_to_device,evidence_for_batch,predict
from ..traffic_neighbors import train_only_neighbours
from ..evaluation import metrics,save_summary
from ..checkpoint import persist_epoch,restore_training,atomic_save

def adapter_data_identity(cfg):
    """S0 depends on fixed formal observed-input and targets, not ckpt alone."""
    root=cache_path(cfg,'formal')
    names=('train_x_obs_norm.npy','val_x_obs_norm.npy',
           'train_history_observed.npy','train_artificial_missing.npy',
           'train_y_norm.npy','val_history_observed.npy','val_artificial_missing.npy','val_y_norm.npy')
    return {name:file_sha256(root/name) for name in names}

def effective_training_batch_size(cfg,smoke=False):
    """Smoke is normally batch=3; optional full-batch mode stresses a single real-size step."""
    if smoke and not bool(cfg.get('runtime',{}).get('smoke_full_batch',False)):
        return 3
    return int(cfg['training']['batch_size'])


def smoke_training_dataset(ds,cfg,smoke=False):
    """Repeat the 24 smoke-only query indices to test full-batch activation memory.

    This is a memory stress test, NOT a throughput or convergence comparison.
    It never loads the formal query cache or changes the formal training protocol.
    """
    if not smoke or not bool(cfg.get('runtime',{}).get('smoke_full_batch',False)):
        return ds
    bs=effective_training_batch_size(cfg,smoke=True)
    if len(ds)==0:raise ValueError('Cannot profile an empty smoke dataset')
    return Subset(ds,[i % len(ds) for i in range(bs)])


def s0_identity(cfg, smoke=False):
    lr, epochs=s0_settings(cfg,smoke)
    return dict(lr=lr, epochs=epochs, weight_decay=0.0001,
                training_batch_size=effective_training_batch_size(cfg,smoke),
                loss='raw_MAE', source_mask_protocol='source_mask_intersection_v2')


def initial_model(cfg,checkpoint,device,adapter_path=None):
    backbone,bc=load_pretrained_stla(Path(cfg['project']['root']),Path(checkpoint),device)
    teacher=MaskAwareSTLAformer(backbone).to(device)
    if adapter_path:
        payload=torch.load(adapter_path,map_location='cpu',weights_only=False)
        if payload.get('version')!=cfg['version']:
            raise ValueError('S0 adapter version mismatch; only fresh v5 artifacts are supported')
        if payload.get('backbone_sha256')!=file_sha256(checkpoint):
            raise ValueError('Adapter trained on a different Backbone checkpoint')
        if payload.get('target')!=cfg['experiment']['target'] or payload.get('dataset')!=cfg['experiment']['dataset']:
            raise ValueError('Adapter target/dataset mismatch')
        if payload.get('s0_identity') != s0_identity(cfg, smoke=bool(payload.get('smoke', False))):
            raise ValueError('S0 adapter training settings mismatch')
        if payload.get('data_identity')!=adapter_data_identity(cfg):
            raise ValueError('S0 adapter data/mask protocol fingerprint mismatch')
        teacher.mask_proj.load_state_dict(payload['mask_proj'],strict=True)
    return teacher,bc

def s0_settings(cfg, smoke=False):
    """Resolve v5 S0 optimizer and epoch parameters, keeping smoke bounded."""
    opts=cfg.get('v5',{})
    return float(opts.get('s0_lr',3e-4)), (1 if smoke else int(opts.get('s0_epochs',8)))


def fit_shared_adapter(cfg,checkpoint,cache,output,device,smoke=False):
    """S0 once per dataset/target/backbone; returns fixed adapter checkpoint."""
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.is_file():
        try:
            initial_model(cfg,checkpoint,device,output)
        except (ValueError,RuntimeError) as exc:
            # Do not silently use an S0 trained under a different model,
            # data protocol, epoch count or optimizer specification.
            print(f'S0_CACHE_INVALIDATED adapter={output} reason={exc}',flush=True)
            output.unlink()
        else:
            print(f'S0_REUSE adapter={output}',flush=True)
            return output
    teacher,bc=initial_model(cfg,checkpoint,device)
    for p in teacher.base.parameters():p.requires_grad_(False)
    for p in teacher.mask_proj.parameters():p.requires_grad_(True)
    train_ds=SupportSplit(cache,'train',24 if smoke else None)
    val_ds=SupportSplit(cache,'val',24 if smoke else None)
    bs=effective_training_batch_size(cfg,smoke)
    tl=_make_loader(smoke_training_dataset(train_ds,cfg,smoke),bs,True,cfg,smoke)
    groups=chronological_partitions(val_ds.window_starts[:len(val_ds)],
        cfg['experiment']['history'],cfg['experiment']['horizon'],smoke=smoke)
    vl=_make_loader(Subset(val_ds,groups['selection']),bs,False,cfg,smoke)
    s0_lr, epochs=s0_settings(cfg,smoke=smoke)
    optimizer=torch.optim.AdamW(teacher.mask_proj.parameters(),lr=s0_lr,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda' and bool(cfg['runtime'].get('amp',True))))
    best_state=copy.deepcopy(teacher.mask_proj.state_dict())
    mean=float(bc['mean']);std=float(bc['std'])
    # Evaluate untouched, zero-initialized Mask Adapter before changing it.
    # If S0 does not improve validation, preserve the original STLA forecast.
    teacher.eval();initial_error=0.;initial_count=0.
    with torch.inference_mode():
        for vb in vl:
            vb=_batch_to_device(vb,device)
            pred,_=teacher(vb['x'],vb['mask'],vb['tod'],vb['dow'])
            v=vb['future_valid'].bool()
            initial_error+=float(((pred.float()*std+mean-vb['y_raw']).abs()*v).sum())
            initial_count+=int(v.sum())
    best=initial_error/max(initial_count,1)
    print(f'S0_BASELINE_VAL_MAE={best:.6f}',flush=True)
    for ep in range(1,epochs+1):
        for training,loader in ((True,tl),(False,vl)):
            teacher.train(training);teacher.base.eval()
            total=0.;count=0
            context=torch.enable_grad() if training else torch.no_grad()
            with context:
                for b in loader:
                    b=_batch_to_device(b,device)
                    if training:optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=scaler.is_enabled()):
                        pred,_=teacher(b['x'],b['mask'],b['tod'],b['dow'])
                        v=b['future_valid'].bool()
                        loss=((pred.float()*std+mean-b['y_raw'].float()).abs()*v).sum()/v.sum().clamp_min(1)
                    if training:
                        if scaler.is_enabled():
                            scaler.scale(loss).backward();scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(teacher.mask_proj.parameters(),5.)
                            scaler.step(optimizer);scaler.update()
                        else:loss.backward();nn.utils.clip_grad_norm_(teacher.mask_proj.parameters(),5.);optimizer.step()
                    total+=float(loss.detach())*int(v.sum());count+=int(v.sum())
            avg=total/max(count,1)
            if not training and avg<best:
                best=avg;best_state=copy.deepcopy(teacher.mask_proj.state_dict())
            print(f'S0_EPOCH ep={ep} phase={"train" if training else "val"} MAE={avg:.6f} best={best:.6f}',flush=True)
    payload=dict(schema=6,version=cfg['version'],target=cfg['experiment']['target'],dataset=cfg['experiment']['dataset'],
                 backbone_sha256=file_sha256(checkpoint),best_val_mae=best,mask_proj=best_state,
                 data_identity=adapter_data_identity(cfg),s0_identity=s0_identity(cfg,smoke),smoke=bool(smoke))
    tmp=output.with_suffix('.partial');torch.save(payload,tmp);os.replace(tmp,output)
    print(f'S0_PASS adapter={output} best_val_MAE={best:.6f}',flush=True)
    return output

