"""Complete v2 RAST training on target-region historical future evidence."""
from __future__ import annotations
import copy,json,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import SupportSplit,FullTrainRepository
from .stla import load_pretrained_stla,MaskSupportSTLAformer
from .model import MaskRASTSTLA
from .memory import build_memory_index
from .graph import graph_neighbours
from .evaluation import metrics,save_summary
from .config import file_sha256
from .checkpoint import persist_epoch,restore_training


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def phase(ep,cfg):
    if cfg['training'].get('schedule')=='frozen':return 'warmup'
    if ep<=int(cfg['training']['warmup_epochs']):return 'warmup'
    if ep<=int(cfg['training']['partial_epochs']):return 'partial'
    return 'full'


def build_model(cfg,checkpoint,device):
    base,ck=load_pretrained_stla(Path(cfg['project']['root']),Path(checkpoint),device)
    stla=MaskSupportSTLAformer(base,int(cfg['encoder']['retrieval_dim']),float(cfg['model']['latent_gate_bias']))
    model=MaskRASTSTLA(stla,int(ck['num_nodes']),int(cfg['experiment']['history']),int(cfg['experiment']['horizon']),cfg).to(device)
    return model,ck


def setup_graph(model,cfg,cache,force=False):
    path=Path(cfg['project']['store_root'])/cfg['experiment']['dataset']/'train_only_graph.npz'
    graph=graph_neighbours(cache,path,int(cfg['retrieval']['graph_neighbors']),
                           int(cfg['retrieval'].get('graph_sampled_rows',1200)),force=force)
    model.set_graph(graph)
    return path


def optimizer(model,cfg):
    bp=list(model.stla.base.parameters());ids={id(p) for p in bp}
    new=[p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{'params':new,'lr':float(cfg['training']['lr_new'])},
                             {'params':bp,'lr':float(cfg['training']['lr_backbone'])}],
                             weight_decay=float(cfg['training']['weight_decay']))


def move(b,device):return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in b.items()}


def make_loader(ds,bs,shuffle,cfg,smoke=False):
    nw=0 if smoke else int(cfg['runtime'].get('num_workers',0))
    kw=dict(batch_size=bs,shuffle=shuffle,num_workers=nw,
            pin_memory=bool(cfg['runtime'].get('pin_memory',True)) and torch.cuda.is_available())
    if nw:kw.update(persistent_workers=bool(cfg['runtime'].get('persistent_workers',True)),
                    prefetch_factor=int(cfg['runtime'].get('prefetch_factor',2)))
    return DataLoader(ds,**kw)


def losses(out,b,mean,std,cfg,mode):
    valid=b['future_valid'].bool();y=b['y_raw'].float();p=out['prediction'].float()*std+mean
    e=p[valid]-y[valid]
    if not e.numel():raise RuntimeError('No valid future labels in batch')
    main=e.abs().mean() if cfg['training'].get('forecast_loss','mae')=='mae' else F.smooth_l1_loss(p[valid],y[valid])
    rmse=torch.sqrt(e.square().mean()+1e-8)
    rank=main.new_zeros(())
    if out['candidate_weights'] is not None and mode!='no_future':
        base=out['base_prediction'].detach();r=out['candidate_residuals'].detach();yn=b['y_norm'].float()
        utility=((base[:,None]-yn[:,None]).abs()-(base[:,None]+r-yn[:,None]).abs()) / max(float(cfg['training'].get('utility_temperature',.25)),1e-6)
        utility=utility.masked_fill(~out['candidate_available'],-1e4)
        targets=torch.softmax(torch.cat([utility,torch.zeros_like(utility[:,:1])],1),1).detach()
        mask=valid[:,None].expand_as(targets)
        rank=-(targets*out['candidate_weights'].float().clamp_min(1e-9).log())[mask].sum()/valid.sum().clamp_min(1)
    total=main+float(cfg['training'].get('rmse_weight',0))*rmse+float(cfg['training'].get('rank_weight',0))*rank
    return total,{'mae':float(e.abs().mean().detach()),'rmse':float(rmse.detach()),
                  'ranking':float(rank.detach()),'gate':float(out['forecast_gate'].mean().detach()),
                  'loss':float(total.detach())}


def run_epoch(model,loader,index,repo,device,mean,std,cfg,mode,opt=None,ep=1,scaler=None):
    training=opt is not None;model.train(training);model.stla.base.eval();cum={};size=0
    with torch.enable_grad() if training else torch.no_grad():
        for step,bb in enumerate(loader,1):
            b=move(bb,device)
            if training:opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=bool(cfg['runtime'].get('amp',False)) and device.type=='cuda'):
                out=model(b,index,repo,mode=mode);loss,parts=losses(out,b,mean,std,cfg,mode)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward();scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training']['grad_clip']))
                    scaler.step(opt);scaler.update()
                else:
                    loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training']['grad_clip']));opt.step()
            bs=int(b['x'].shape[0]);size+=bs
            for k,v in parts.items():cum[k]=cum.get(k,0)+v*bs
            if training and step%int(cfg['runtime']['print_every'])==0:
                print(f'epoch={ep:03d} batch={step:05d}/{len(loader):05d} MAE={cum["mae"]/size:.5f} rank={cum["ranking"]/size:.5f}',flush=True)
    return {k:v/max(1,size) for k,v in cum.items()}


def predict(model,loader,index,repo,device,mean,std,mode='full',amp=False):
    arrays={k:[] for k in ('prediction','base','true','valid','direct','correction')};g=[];model.eval()
    with torch.no_grad():
        for bb in loader:
            b=move(bb,device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp and device.type=='cuda'):
                o=model(b,index,repo,mode=mode)
            arrays['prediction'].append((o['prediction'].float()*std+mean).cpu().numpy())
            arrays['base'].append((o['base_prediction'].float()*std+mean).cpu().numpy())
            arrays['true'].append(b['y_raw'].cpu().numpy());arrays['valid'].append(b['future_valid'].cpu().numpy())
            arrays['direct'].append(b['direct_mask'].cpu().numpy());arrays['correction'].append((o['correction'].float()*std).cpu().numpy())
            g.append(float(o['forecast_gate'].mean()))
    result={k:np.concatenate(v) for k,v in arrays.items()};result['gate_mean']=float(np.mean(g));return result


def train(cfg,cache,checkpoint,run_dir,smoke=False,resume=None):
    seed_all(int(cfg['experiment']['seed']));dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,ck=build_model(cfg,checkpoint,dev)
    graph_path=setup_graph(model,cfg,cache)
    mean=float(ck['mean']);std=float(ck['std']);limit=24 if smoke else None
    tr=SupportSplit(cache,'train',limit);va=SupportSplit(cache,'val',limit);repo=FullTrainRepository(cache,256 if smoke else None)
    if not smoke and cfg['runtime'].get('preload_repository_gpu',True):repo.preload(dev)
    bs=3 if smoke else int(cfg['training']['batch_size'])
    tl=make_loader(tr,bs,True,cfg,smoke);vl=make_loader(va,bs,False,cfg,smoke)
    mode=cfg['training'].get('ablation','full');opt=optimizer(model,cfg)
    scaler=torch.amp.GradScaler('cuda',enabled=bool(cfg['runtime'].get('amp',False)) and dev.type=='cuda')
    index=None;best=float('inf');best_ep=0;stale=0;prev=None;history=[];start_epoch=1
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    if resume:
        restored=restore_training(resume,model,opt,scaler,cfg,checkpoint)
        start_epoch=int(restored['epoch'])+1;best=float(restored['best_val_mae']);best_ep=int(restored['best_epoch'])
        stale=int(restored['stale']);prev=restored['previous_phase'];history=list(restored['history'])
    maxep=2 if smoke else int(cfg['training']['max_epochs']);interval=int(cfg['retrieval']['store_refresh_epochs'])
    print(f'PERF_CONFIG model=RAST batch={bs} amp={scaler.is_enabled()} repository_gpu={repo._device}',flush=True)
    for ep in range(start_epoch,maxep+1):
        ph=phase(ep,cfg);model.set_train_phase(ph,int(cfg['training']['partial_last_layers']))
        if ph!=prev:stale=0;prev=ph;print(f'TRAIN_PHASE epoch={ep} phase={ph}',flush=True)
        if index is None or ((ep-1)%interval==0):
            t0=time.time();index=build_memory_index(model.memory_encoder,repo,dev,
                                                      16 if smoke else int(cfg['retrieval']['store_batch_size']));index.to(dev)
            print(f'MEMORY_REFRESH epoch={ep} time={time.time()-t0:.1f}s',flush=True)
        if dev.type=='cuda':torch.cuda.reset_peak_memory_stats(dev)
        t0=time.time();a=run_epoch(model,tl,index,repo,dev,mean,std,cfg,mode,opt,ep,scaler)
        train_seconds=time.time()-t0
        val_index=build_memory_index(model.memory_encoder,repo,dev,
                   16 if smoke else int(cfg['retrieval']['store_batch_size']))
        val_index.to(dev)
        v=run_epoch(model,vl,val_index,repo,dev,mean,std,cfg,mode,None,ep,scaler)
        val_seconds=time.time()-t0-train_seconds
        rec={'epoch':ep,'phase':ph,**{f'train_{k}':x for k,x in a.items()},
             **{f'val_{k}':x for k,x in v.items()},'seconds':time.time()-t0,
             'train_seconds':train_seconds,'val_seconds':val_seconds,
             'train_samples_per_second':len(tr)/max(train_seconds,1e-5),
             'peak_gpu_gib':torch.cuda.max_memory_reserved(dev)/1024**3 if dev.type=='cuda' else 0.0}
        history.append(rec);print('EPOCH '+json.dumps(rec),flush=True)
        improved=v['mae'] < best - float(cfg['training']['min_delta'])
        if improved:best=v['mae'];best_ep=ep;stale=0
        else:stale+=1
        pd.DataFrame(history).to_csv(run_dir/'history.csv',index=False)
        persist_epoch(run_dir,model,opt,scaler,cfg,checkpoint,ck,'Mask_RAST-STLA',ep,best,best_ep,history,stale,ph,improved)
        if improved:
            val_index.save(run_dir/'best_memory_index.npz',{
                'version':cfg['version'],'memory_rows':len(repo),'dataset':cfg['experiment']['dataset'],
                'target':cfg['experiment']['target'],'graph':str(graph_path)})
        if stale>=int(cfg['training']['patience']) and (ph=='full' or cfg['training'].get('schedule')=='frozen'):
            print(f'EARLY_STOP epoch={ep} best_epoch={best_ep}',flush=True);break
    best_path=run_dir/'best.pt'
    if not best_path.is_file():raise RuntimeError('No completed best checkpoint')
    saved=torch.load(best_path,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True);model.to(dev)
    index=build_memory_index(model.memory_encoder,repo,dev,16 if smoke else int(cfg['retrieval']['store_batch_size']))
    index.to(dev);index.save(run_dir/'best_memory_index.npz',{'version':cfg['version'],'memory_rows':len(repo),
        'dataset':cfg['experiment']['dataset'],'target':cfg['experiment']['target'],'graph':str(graph_path)})
    pred=predict(model,vl,index,repo,dev,mean,std,mode,amp=scaler.is_enabled())
    vm=metrics(pred['prediction'],pred['true'],pred['valid'],pred['direct'])
    bm=metrics(pred['base'],pred['true'],pred['valid'],pred['direct'])
    np.savez_compressed(run_dir/'val_predictions.npz',**{k:v.astype(np.float32) for k,v in pred.items() if isinstance(v,np.ndarray)})
    save_summary(run_dir/'summary.json',{'model':'Mask_RAST-STLA-v2.1','best_epoch':best_ep,
        'validation':vm,'base_validation':bm,'gate_mean':pred['gate_mean'],
        'backbone_checkpoint':str(checkpoint),'ablation':mode,'test_evaluated':False})
    print(f'TRAIN_PASS run_dir={run_dir} best_epoch={best_ep} val_MAE={vm["MAE"]:.6f}',flush=True)
    return run_dir/'best.pt'
