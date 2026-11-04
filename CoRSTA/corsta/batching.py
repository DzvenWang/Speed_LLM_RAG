"""OOM-safe full model batch probe. Priority: 64 -> 128 (headroom permitting) -> 32 -> 16.
Only sizes in {16,32,64,128} are considered for CUDA training. No silent CPU fallback.
"""
from __future__ import annotations
import copy,gc,time
import torch
from torch.utils.data._utils.collate import default_collate

CHOICES=(16,32,64,128)


def sample_batch(dataset,size,device):
    row=dataset[0]
    out={}
    for k,v in row.items():
        if isinstance(v,torch.Tensor):
            a=v.unsqueeze(0)
            out[k]=a.expand(size,*v.shape).contiguous().to(device)
        elif isinstance(v,(int,float)):
            out[k]=torch.full((size,),v,device=device)
        else:raise TypeError(f'Unsupported batch sample key {k}: {type(v)}')
    return out


def is_oom(exc):
    if isinstance(exc,torch.cuda.OutOfMemoryError):return True
    return isinstance(exc,RuntimeError) and 'out of memory' in str(exc).lower()


def probe(model,dataset,size,device,amp=True,capacity_fraction=.85):
    """Uses *real* forward and backward incl. all branches; does not step weights.

    Additional capacity margin accounts for Adam state, variable data and fragmentation.
    Candidate probe is performed at the selected training stage, with frozen parameters
    already configured. Output includes peak allocated, physical GPU capacity.
    """
    if device.type!='cuda':
        return {'size':size,'ok':True,'reason':'CPU no VRAM profiling','peak_gib':0}
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    initial=torch.cuda.memory_allocated(device)
    total=torch.cuda.get_device_properties(device).total_memory
    try:
        b=sample_batch(dataset,size,device)
        model.train()
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
            o=model(b)
            loss=o['refined'].float().square().mean()+0.01*o['c1_posthoc'].float().square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite probe loss')
        loss.backward()
        torch.cuda.synchronize(device)
        peak=torch.cuda.max_memory_allocated(device)
        # Keep a substantial margin for optimizer states, dataloader and fragmentation.
        ok=(peak+max(0.5*1024**3,0.12*total)) <= capacity_fraction*total
        return {'size':size,'ok':ok,'reason':'pass' if ok else 'insufficient headroom',
                'peak_gib':round(peak/1024**3,3),'gpu_gib':round(total/1024**3,3),
                'headroom_gib':round((total-peak)/1024**3,3)}
    except (RuntimeError,MemoryError) as e:
        if not is_oom(e):raise
        return {'size':size,'ok':False,'reason':'CUDA OOM: '+str(e)[:180], 'peak_gib':None}
    finally:
        model.zero_grad(set_to_none=True)
        if 'b' in locals():del b
        if 'o' in locals():del o
        if 'loss' in locals():del loss
        gc.collect();torch.cuda.empty_cache()


def select_batch(model,dataset,device,amp=True,initial=64,capacity_fraction=.85,logger=print):
    if initial not in CHOICES:raise ValueError(f'Initial size must be in {CHOICES}')
    if device.type!='cuda':
        logger(f'CORSTA_BATCH_AUTO_CPU selected={initial} reason=CPU has no GPU capacity measurement')
        return initial,[]
    # Test preferred size first. Upsize only when preferred FITS and has spare
    # memory; on failure, NEVER try a larger batch before smaller choices.
    attempts=[];chosen=None
    first=probe(model,dataset,initial,device,amp,capacity_fraction)
    attempts.append(first)
    logger(f'CORSTA_BATCH_PROBE size={initial} success={first["ok"]} peak_gib={first.get("peak_gib")} reason={first["reason"]}')
    if first['ok']:
        chosen=initial
        larger=[n for n in CHOICES if n>initial]
        if larger and first['peak_gib'] <= first['gpu_gib']*.58:
            next_size=min(larger)
            r=probe(model,dataset,next_size,device,amp,capacity_fraction)
            attempts.append(r)
            logger(f'CORSTA_BATCH_PROBE size={next_size} success={r["ok"]} peak_gib={r.get("peak_gib")} reason={r["reason"]}')
            if r['ok']:chosen=next_size
    else:
        for size in sorted((n for n in CHOICES if n<initial),reverse=True):
            r=probe(model,dataset,size,device,amp,capacity_fraction)
            attempts.append(r)
            logger(f'CORSTA_BATCH_PROBE size={size} success={r["ok"]} peak_gib={r.get("peak_gib")} reason={r["reason"]}')
            if r['ok']:
                chosen=size;break
    if chosen is None:
        raise RuntimeError(f'No safe batch size in {CHOICES}; attempts={attempts}')
    logger(f'CORSTA_BATCH_SELECTED batch={chosen} initial={initial} alternatives={CHOICES}')
    return chosen,attempts


def smaller_size(size):
    available=[n for n in CHOICES if n<size]
    return max(available) if available else None
