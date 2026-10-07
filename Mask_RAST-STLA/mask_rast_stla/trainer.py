from __future__ import annotations
import copy,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader
from .data import SupportSplit,FullTrainRepository
from .stla import load_pretrained_stla,MaskSupportSTLAformer
from .model import MaskRASTSTLA
from .memory import build_memory_index
from .evaluation import metrics,save_summary

def seed_all(seed):
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def phase(ep,cfg):
    if ep<=int(cfg['training']['warmup_epochs']):return 'warmup'
    if ep<=int(cfg['training']['partial_epochs']):return 'partial'
    return 'full'

def build_model(cfg,checkpoint,device):
    base,ck=load_pretrained_stla(Path(cfg['project']['root']),Path(checkpoint),device);ec=cfg['encoder'];stla=MaskSupportSTLAformer(base,int(ec['retrieval_dim']),float(cfg['model']['latent_gate_bias']));m=MaskRASTSTLA(stla,int(ck['num_nodes']),int(cfg['experiment']['history']),int(cfg['experiment']['horizon']),cfg).to(device);return m,ck

def optimizer(model,cfg):
    bp=list(model.stla.base.parameters());ids={id(p) for p in bp};new=[p for p in model.parameters() if id(p) not in ids];return torch.optim.AdamW([{'params':new,'lr':float(cfg['training']['lr_new'])},{'params':bp,'lr':float(cfg['training']['lr_backbone'])}],weight_decay=float(cfg['training']['weight_decay']))

def move(b,d):return {k:(v.to(d,non_blocking=True) if torch.is_tensor(v) else v) for k,v in b.items()}
def raw_loss(p,y,v,mean,std):
    pr=p*std+mean;m=v.bool();return nn.functional.smooth_l1_loss(pr[m],y[m])
def multi_positive_nce(q,k,g,temp):
    q=nn.functional.normalize(q,dim=-1);k=nn.functional.normalize(k,dim=-1);log=q@k.t()/max(float(temp),1e-6);pos=g[:,None].eq(g[None,:]);num=torch.logsumexp(log.masked_fill(~pos,-float('inf')),dim=1);den=torch.logsumexp(log,dim=1);return -(num-den).mean()

def _make_loader(ds,batch_size,shuffle,cfg,smoke=False):
    workers=0 if smoke else int(cfg['runtime'].get('num_workers',0)); pin=bool(cfg['runtime'].get('pin_memory',True)) and torch.cuda.is_available()
    kw=dict(batch_size=int(batch_size),shuffle=shuffle,num_workers=workers,pin_memory=pin)
    if workers>0:
        kw['persistent_workers']=bool(cfg['runtime'].get('persistent_workers',True)); kw['prefetch_factor']=int(cfg['runtime'].get('prefetch_factor',2))
    return DataLoader(ds,**kw)

def run_epoch(model,loader,index,repo,device,mean,std,cfg,opt=None,ep=1):
    tr=opt is not None;model.train(tr);acc={'forecast':0.,'align':0.,'retrieval':0.,'gate':0.};n=0;ctx=torch.enable_grad() if tr else torch.no_grad()
    with ctx:
        for bi,b in enumerate(loader,1):
            b=move(b,device)
            if tr:opt.zero_grad(set_to_none=True)
            o=model(b,index,repo);main=raw_loss(o['prediction'],b['y_raw'].float(),b['future_valid'],mean,std);al=(multi_positive_nce(o['q_temporal'],o['k_temporal'],b['group_id'].long(),cfg['training']['alignment_temperature'])+multi_positive_nce(o['q_spatial'],o['k_spatial'],b['group_id'].long(),cfg['training']['alignment_temperature']))/2;ra=raw_loss(o['retrieval_prediction'],b['y_raw'].float(),b['future_valid'],mean,std);gr=o['forecast_gate'].mean()+(o['latent_gate'].mean() if o['latent_gate'] is not None else 0.0);loss=main+float(cfg['training']['alignment_weight'])*al+float(cfg['training']['retrieval_aux_weight'])*ra+float(cfg['training']['gate_reg_weight'])*gr
            if tr:loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training']['grad_clip']));opt.step()
            bs=b['x'].shape[0];n+=bs;acc['forecast']+=float(main.detach())*bs;acc['align']+=float(al.detach())*bs;acc['retrieval']+=float(ra.detach())*bs;acc['gate']+=float(torch.as_tensor(gr).detach())*bs
            if tr and bi%int(cfg['runtime']['print_every'])==0:print(f'epoch={ep:03d} batch={bi:05d}/{len(loader):05d} forecast={acc["forecast"]/n:.6f} align={acc["align"]/n:.6f} retr={acc["retrieval"]/n:.6f}',flush=True)
    return {k:v/max(n,1) for k,v in acc.items()}

def predict(model,loader,index,repo,device,mean,std):
    ps=[];ys=[];vs=[];ds=[];fg=[];lg=[];model.eval()
    with torch.no_grad():
        for b in loader:
            b=move(b,device);o=model(b,index,repo);ps.append((o['prediction']*std+mean).cpu().numpy());ys.append(b['y_raw'].cpu().numpy());vs.append(b['future_valid'].cpu().numpy());ds.append(b['direct_mask'].cpu().numpy());fg.append(float(o['forecast_gate'].mean()));lg.append(float(o['latent_gate'].mean()) if o['latent_gate'] is not None else 0.)
    return np.concatenate(ps),np.concatenate(ys),np.concatenate(vs),np.concatenate(ds),float(np.mean(fg)),float(np.mean(lg))

def train(cfg,cache,checkpoint,run_dir,smoke=False):
    seed_all(int(cfg['experiment']['seed']));device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');model,bc=build_model(cfg,checkpoint,device);mean=float(bc['mean']);std=float(bc['std']);limit=32 if smoke else None;tr=SupportSplit(cache,'train',limit);va=SupportSplit(cache,'val',limit);repo=FullTrainRepository(cache,256 if smoke else None)
    if not smoke and device.type=='cuda' and bool(cfg['runtime'].get('preload_repository_gpu',True)): repo.preload(device)
    bs=4 if smoke else int(cfg['training']['batch_size']);trl=_make_loader(tr,bs,True,cfg,smoke);vl=_make_loader(va,bs,False,cfg,smoke);print(f'RUNTIME batch_size={bs} train_batches={len(trl)} val_batches={len(vl)} num_workers={0 if smoke else int(cfg["runtime"].get("num_workers",0))} pin_memory={bool(cfg["runtime"].get("pin_memory",True))} repo_gpu={repo._device is not None}',flush=True)
    opt=optimizer(model,cfg);best=float('inf');state=None;be=0;stale=0;hist=[];maxep=2 if smoke else int(cfg['training']['max_epochs']);index=None
    for ep in range(1,maxep+1):
        ph=phase(ep,cfg);model.set_train_phase(ph,int(cfg['training']['partial_last_layers']))
        refresh_seconds=0.0
        if index is None or ep==1 or (ep-1)%int(cfg['retrieval']['store_refresh_epochs'])==0:
            print(f'MEMORY_REFRESH epoch={ep}',flush=True);rt=time.time();index=build_memory_index(model.memory_encoder,repo,device,16 if smoke else int(cfg['retrieval']['store_batch_size']));index.to(device);refresh_seconds=time.time()-rt;print(f'MEMORY_REFRESH_DONE epoch={ep} seconds={refresh_seconds:.3f}',flush=True)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
        t=time.time();a=run_epoch(model,trl,index,repo,device,mean,std,cfg,opt,ep);v=run_epoch(model,vl,index,repo,device,mean,std,cfg,None,ep)
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed=time.time()-t;throughput=(len(tr)+len(va))/max(elapsed,1e-9);rec={'epoch':ep,'phase':ph,**{f'train_{k}':x for k,x in a.items()},**{f'val_{k}':x for k,x in v.items()},'seconds':elapsed,'memory_refresh_seconds':refresh_seconds,'samples_per_sec':throughput}
        if device.type=='cuda':rec.update({'gpu_peak_allocated_gb':torch.cuda.max_memory_allocated(device)/1024**3,'gpu_peak_reserved_gb':torch.cuda.max_memory_reserved(device)/1024**3})
        hist.append(rec);print('EPOCH '+json.dumps(rec),flush=True);score=v['forecast']
        if score<best-float(cfg['training']['min_delta']):best=score;be=ep;stale=0;state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            stale+=1
            if stale>=int(cfg['training']['patience']):break
    if state is None:raise RuntimeError('No checkpoint selected')
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True);pd.DataFrame(hist).to_csv(run_dir/'history.csv',index=False);model.load_state_dict(state);model.to(device);index=build_memory_index(model.memory_encoder,repo,device,16 if smoke else int(cfg['retrieval']['store_batch_size']));index.to(device);index.save(run_dir/'best_memory_index.npz',{'best_epoch':be,'memory_rows':len(repo),'encoder':'full-train asymmetric memory encoder'});p,y,v,d,fg,lg=predict(model,vl,index,repo,device,mean,std);vm=metrics(p,y,v,d);torch.save({'version':cfg['version'],'model':'Mask_RAST-STLA','target':cfg['experiment']['target'],'seed':cfg['experiment']['seed'],'state_dict':state,'backbone_checkpoint':str(checkpoint),'backbone_model_config':bc['model_config'],'num_nodes':bc['num_nodes'],'mean':mean,'std':std,'best_epoch':be,'best_val_loss':best,'config':cfg},run_dir/'best.pt');np.savez_compressed(run_dir/'val_predictions.npz',prediction=p.astype(np.float32));save_summary(run_dir/'summary.json',{'model':'Mask_RAST-STLA','best_epoch':be,'validation':vm,'forecast_gate_mean':fg,'latent_gate_mean':lg,'backbone_checkpoint':str(checkpoint),'test_evaluated':False});print(f'TRAIN_PASS run_dir={run_dir} best_epoch={be} val_MAE={vm["MAE"]:.6f}',flush=True);return run_dir/'best.pt'
