"""Atomic per-epoch checkpoints for Gadi walltime-safe v2.3 Formal jobs."""
from __future__ import annotations
import os, random
from pathlib import Path
import numpy as np
import torch
from .config import file_sha256


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.writing')
    try:
        torch.save(obj,temp)
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def checkpoint_core(model,cfg,backbone,bc,model_name,best,best_epoch):
    return {'version':cfg['version'],'model':model_name,
            'target':cfg['experiment']['target'],'seed':cfg['experiment']['seed'],
            'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'backbone_checkpoint':str(backbone),'backbone_sha256':file_sha256(backbone),
            'num_nodes':bc['num_nodes'],'backbone_model_config':bc['model_config'],
            'mean':float(bc['mean']),'std':float(bc['std']),
            'best_epoch':best_epoch,'best_val_mae':best,'config':cfg,
            'adapter_sha256':file_sha256(cfg['runtime']['teacher_adapter_checkpoint'])
                if str(cfg.get('version','')).startswith(('4.','5.')) and cfg.get('runtime',{}).get('teacher_adapter_checkpoint') else None}


def persist_epoch(run_dir,model,opt,scaler,cfg,backbone,bc,model_name,epoch,best,best_epoch,history,stale,previous_phase,improved):
    core=checkpoint_core(model,cfg,backbone,bc,model_name,best,best_epoch)
    # Emit completed best checkpoint before last; both files are atomic.
    if improved:atomic_save(core,Path(run_dir)/'best.pt')
    last={**core,'epoch':epoch,'optimizer_state_dict':opt.state_dict(),
          'scaler_state_dict':scaler.state_dict() if scaler is not None else None,
          'history':history,'stale':stale,'previous_phase':previous_phase,
          'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
          'rng_torch':torch.random.get_rng_state(),
          'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    atomic_save(last,Path(run_dir)/'last.pt')
    print(f'CHECKPOINT_SAVED epoch={epoch} best_epoch={best_epoch} best_val_mae={best:.6f} folder={run_dir}',flush=True)


def restore_training(path,model,opt,scaler,cfg,backbone):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('version')!=cfg['version']:
        raise RuntimeError('Resume only compatible with exactly the same version')
    if ck.get('backbone_sha256')!=file_sha256(backbone):
        raise RuntimeError('Resume backbone SHA256 mismatch')
    old=ck['config']
    for section,keys in [('experiment',['dataset','target','seed']),
                         ('training',['batch_size','ablation','schedule','max_epochs',
                                      'warmup_epochs','partial_epochs'])]:
        for key in keys:
            if old[section].get(key)!=cfg[section].get(key):
                raise RuntimeError(f'Resume config mismatch: {section}.{key}')
    if old.get('retrieval')!=cfg.get('retrieval') or old.get('runtime',{}).get('frozen_replay_teacher')!=cfg.get('runtime',{}).get('frozen_replay_teacher'):
        raise RuntimeError('Resume retrieval configuration mismatch')
    model.load_state_dict(ck['state_dict'],strict=True)
    opt.load_state_dict(ck['optimizer_state_dict'])
    if scaler is not None and ck.get('scaler_state_dict') is not None:
        scaler.load_state_dict(ck['scaler_state_dict'])
    random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy']);torch.random.set_rng_state(ck['rng_torch'])
    if torch.cuda.is_available() and ck.get('rng_cuda') is not None:torch.cuda.set_rng_state_all(ck['rng_cuda'])
    print(f'RESUME_OK start_epoch={ck["epoch"]+1} best_epoch={ck["best_epoch"]}',flush=True)
    return ck
