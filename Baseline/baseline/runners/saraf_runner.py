#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
sys.path.insert(0, str(ROOT))

from baseline.runners.common import (  # noqa: E402
    SARAFTensorDataset,
    build_time_marks,
    make_run_dir,
    masked_mse,
    require_gpu,
    save_val_outputs,
    scale_stats,
    set_seed,
    simple_namespace,
    validate_stage,
)


def import_model():
    vendor = ROOT / 'vendor' / 'SARAF'
    sys.path.insert(0, str(vendor))
    try:
        from models.SARAF import Model
    except Exception as e:
        raise RuntimeError(f'SARAF official import failed from {vendor}: {e!r}') from e
    return Model


def overlap_masker(query_starts: np.ndarray, memory_starts: np.ndarray, span: int):
    """Replace stock contiguous-index masking with Protocol-time overlap masking.

    A Train query must not retrieve any Train-memory window whose history/future
    interval overlaps the query history/future interval. This is stricter than
    self-exclusion and remains correct after balanced Protocol sampling/reordering.
    """
    q = torch.as_tensor(query_starts, dtype=torch.long)
    m = torch.as_tensor(memory_starts, dtype=torch.long)

    def masked_sim(self, sim, index, bsz, x):
        idx_cpu = index.detach().long().cpu()
        qs = q[idx_cpu].to(x.device)
        ms = m.to(x.device)
        # intervals [start, start+span); overlap iff ms < qs+span and qs < ms+span
        overlap = (ms.unsqueeze(0) < (qs + span).unsqueeze(1)) & (qs.unsqueeze(1) < (ms + span))
        return sim.masked_fill(overlap.unsqueeze(0), float('-inf'))

    return masked_sim


def _probe_time_bonus(rt, query_ds, device) -> dict:
    """Fail loudly if the time-aware term is constant after timestamp adaptation."""
    n = min(8, len(query_ds))
    if n <= 0:
        raise RuntimeError('SARAF time-bonus probe has no query rows')
    q = torch.from_numpy(np.asarray(query_ds.xm[:n], dtype=np.float32)).to(device)
    with torch.no_grad():
        bonus = rt.compute_time_aware_bonus(q, rt.train_data_all_mark, device)
    stats = {
        'min': float(bonus.min().item()),
        'max': float(bonus.max().item()),
        'mean': float(bonus.mean().item()),
        'std': float(bonus.std().item()),
    }
    print('SARAF TIME BONUS PROBE ' + json.dumps(stats, sort_keys=True), flush=True)
    if stats['max'] - stats['min'] < 1e-5 or stats['std'] < 1e-6:
        raise RuntimeError(
            'SARAF time-aware bonus is effectively constant. Check time_index parsing; '
            'numeric row-index columns must not be interpreted as datetimes.'
        )
    return stats


def precompute_retrieval(model, memory_ds, train_query_ds, val_query_ds, cfg, device):
    rt = model.rt
    # Official implementation has no 5-minute map; fix only frequency metadata,
    # leaving official similarity/MMR/stationarity algorithms untouched.
    rt.steps_per_day = 288
    rt.steps_per_year = 288 * 365
    rt.prepare_dataset(memory_ds)
    time_bonus_probe = _probe_time_bonus(rt, val_query_ds, device)

    H = int(cfg['experiment']['history']); P = int(cfg['experiment']['horizon'])
    span = H + P
    memory_starts = memory_ds.meta.window_start.astype(int).to_numpy()
    query_starts = train_query_ds.meta.window_start.astype(int).to_numpy()
    rt.masked_sim = MethodType(overlap_masker(query_starts, memory_starts, span), rt)

    print('SARAF RETRIEVAL train queries -> Train-only full-history memory (time-overlap excluded)', flush=True)
    train_rt = rt.retrieve_all(train_query_ds, train=True, device=device)
    val_starts = val_query_ds.meta.window_start.astype(int).to_numpy()
    rt.masked_sim = MethodType(overlap_masker(val_starts, memory_starts, span), rt)
    print('SARAF RETRIEVAL val queries -> Train-only full-history memory (time-overlap excluded)', flush=True)
    # train=True is intentional here: it activates only our external overlap mask;
    # retrieval generation is otherwise identical and no query target is exposed.
    val_rt = rt.retrieve_all(val_query_ds, train=True, device=device)
    model.retrieval_temporal_dict = {
        'train': train_rt.detach().to(device, non_blocking=True),
        'valid': val_rt.detach().to(device, non_blocking=True),
    }
    # Drop raw retrieval store after precompute just like official prepare_dataset.
    del model.rt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        'train_retrieval_shape': list(train_rt.shape),
        'val_retrieval_shape': list(val_rt.shape),
        'memory_rows': len(memory_ds),
        'train_query_rows': len(train_query_ds),
        'val_query_rows': len(val_query_ds),
        'overlap_exclusion_span_steps': span,
        'time_bonus_probe': time_bonus_probe,
    }


def loader(ds, batch: int, shuffle: bool):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True, drop_last=False)


def run_epoch(model, dl, device, mode: str, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0; denom = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for idx, x, y, xm, _ym in dl:
            idx = idx.long().to(device, non_blocking=True)
            x = x.float().to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            xm = xm.float().to(device, non_blocking=True)
            ii = idx.detach().cpu().numpy()
            valid_np = np.asarray(dl.dataset.valid[ii], np.uint8)
            valid = torch.from_numpy(valid_np).bool().to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x, xm, idx, mode=mode)
            if pred.shape != y.shape:
                raise RuntimeError(f'SARAF output contract failed: pred={tuple(pred.shape)} y={tuple(y.shape)}')
            loss = masked_mse(pred, y, valid)
            if train:
                loss.backward(); optimizer.step()
            n = int(valid.sum().item()); total += float(loss.detach().item()) * max(n,1); denom += max(n,1)
    return total / max(denom,1)


def predict(model, dl, device, mean: float, std: float) -> np.ndarray:
    out=[]; model.eval()
    with torch.no_grad():
        for idx,x,_y,xm,_ym in dl:
            p=model(x.float().to(device), xm.float().to(device), idx.long().to(device), mode='valid')
            out.append(p.detach().cpu().numpy())
    pn=np.concatenate(out,axis=0).astype(np.float32)
    return pn*np.float32(std)+np.float32(mean)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage', default='development', choices=['development','formal'])
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--smoke', action='store_true')
    args=ap.parse_args()

    cfg,cache=validate_stage(args.stage,'SARAF'); rcfg=cfg['runners']['SARAF']
    seed=int(cfg['experiment']['seed'] if args.seed is None else args.seed); set_seed(seed); device=require_gpu()
    txm,tym,tsrc=build_time_marks(cfg,cache,'train'); vxm,vym,_=build_time_marks(cfg,cache,'val')
    # Retrieval memory has complete historical Train signals. Query streams retain exact frozen incomplete observations.
    lim_mem=256 if args.smoke else None; lim_train=64 if args.smoke else None; lim_val=64 if args.smoke else None
    memory_ds=SARAFTensorDataset(cache,'train',txm,tym,query_signal='full',limit=lim_mem)
    train_ds=SARAFTensorDataset(cache,'train',txm,tym,query_signal='obs',limit=lim_train)
    val_ds=SARAFTensorDataset(cache,'val',vxm,vym,query_signal='obs',limit=lim_val)
    N=int(train_ds.x.shape[2]); H=int(cfg['experiment']['history']); P=int(cfg['experiment']['horizon'])

    Model=import_model()
    model_cfg=simple_namespace(
        task_name='long_term_forecast', seq_len=H, pred_len=P,
        features=str(rcfg.get('features','M')), enc_in=N,
        n_period=int(rcfg.get('n_period',1)), topm=int(rcfg.get('topm',20)),
        freq=str(rcfg.get('freq','5min')), time_aware_weight=float(rcfg.get('time_aware_weight',0.3)),
        fusion_method=str(rcfg.get('fusion_method','weighted')), use_gpu=True, gpu=0,
    )
    model=Model(model_cfg).to(device)
    nparams=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'SARAF START stage={args.stage} seed={seed} device={device} params={nparams} H={H} P={P} N={N}',flush=True)
    print('SARAF SAFETY Train-only retrieval memory; no Test dataset instantiated.',flush=True)

    retrieval_info=precompute_retrieval(model,memory_ds,train_ds,val_ds,cfg,device)
    batch=int(rcfg['batch_size']); train_dl=loader(train_ds,batch,True); val_dl=loader(val_ds,batch,False)
    optim=torch.optim.Adam(model.parameters(),lr=float(rcfg['learning_rate']))

    if args.smoke:
        idx,x,y,xm,_=next(iter(train_dl)); optim.zero_grad(set_to_none=True)
        p=model(x.float().to(device),xm.float().to(device),idx.long().to(device),mode='train')
        yy=y.float().to(device); ii=idx.detach().cpu().numpy(); vv=torch.from_numpy(np.asarray(train_ds.valid[ii],np.uint8)).bool().to(device)
        loss=masked_mse(p,yy,vv); loss.backward(); optim.step()
        vi,vx,vy,vxm0,_=next(iter(val_dl)); model.eval()
        with torch.no_grad(): vp=model(vx.float().to(device),vxm0.float().to(device),vi.long().to(device),mode='valid')
        print(f'SARAF GPU SMOKE PASS train_loss={loss.item():.6f} output={tuple(vp.shape)} retrieval={retrieval_info}',flush=True)
        return

    max_epochs=int(rcfg['max_epochs']); patience=int(rcfg.get('patience',5)); best=float('inf'); best_state=None; best_epoch=0; stale=0; t0=time.time()
    for epoch in range(1,max_epochs+1):
        et=time.time(); tr=run_epoch(model,train_dl,device,'train',optim); va=run_epoch(model,val_dl,device,'valid',None)
        print(f'EPOCH {epoch:03d} train_MSE_norm={tr:.6f} val_MSE_norm={va:.6f} lr={optim.param_groups[0]["lr"]:.8g} sec={time.time()-et:.1f}',flush=True)
        if va < best-1e-10:
            best=va; best_epoch=epoch; stale=0; best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            stale+=1
            if stale>=patience:
                print(f'EARLY STOP epoch={epoch} best_epoch={best_epoch}',flush=True); break
    if best_state is None: raise RuntimeError('SARAF did not produce a checkpoint')
    model.load_state_dict(best_state); model.to(device)
    mean,std=scale_stats(cache); pred_raw=predict(model,val_dl,device,mean,std)
    run_dir=make_run_dir(cfg,'SARAF',args.stage,seed)
    torch.save({'model':best_state,'best_epoch':best_epoch,'best_val_mse_norm':best},run_dir/'best.pt')
    summary=save_val_outputs(cfg,cache,run_dir,pred_raw,'SARAF',seed,args.stage,{
        'best_epoch':int(best_epoch),'best_val_MSE_norm':float(best),'params':int(nparams),'wall_seconds':float(time.time()-t0),
        'official_architecture':'vendor/SARAF/models/SARAF.py::Model','official_retrieval':'vendor/SARAF/layers/Retrieval.py::RetrievalTool',
        'time_feature_source':tsrc,'frequency_override':'5min=>288_steps_per_day','retrieval':retrieval_info,
        'retrieval_memory_split':'train','retrieval_memory_signal':'full_train_history','query_signal':'incomplete_history',
        'train_overlap_mask':'Protocol window_start interval exclusion','checkpoint_selection':'val_only','test_access':False,
    })
    print('SARAF COMPLETE',json.dumps({'run_dir':str(run_dir),'val':summary['val']},default=str),flush=True)

if __name__=='__main__': main()
