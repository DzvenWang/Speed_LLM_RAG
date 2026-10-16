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
from .model import PCFTSTLA
from .memory import build_memory_index
from .graph import graph_neighbours
from .propagation import build_directional_graph
from .seasonal import train_future_reference
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
    model=PCFTSTLA(stla,int(ck['num_nodes']),int(cfg['experiment']['history']),int(cfg['experiment']['horizon']),cfg).to(device)
    return model,ck


def setup_graph(model,cfg,cache,force=False):
    path=Path(cfg['project']['store_root'])/cfg['experiment']['dataset']/'train_only_statistical_graph_v1.npz'
    graph=graph_neighbours(cache,path,int(cfg['retrieval']['graph_neighbors']),
                           int(cfg['retrieval'].get('graph_sampled_rows',1200)),force=force,
                           lag_steps=cfg['retrieval'].get('graph_lag_steps',[0,1,2,3]))
    model.set_graph(graph)
    if model.propagation_enabled:
        weights,conf=build_directional_graph(cache,graph,
            Path(cfg['project']['store_root'])/cfg['experiment']['dataset']/'train_only_directional_v1.npz',
            lags=cfg.get('propagation',{}).get('lags',[1,2,3,4,5,6]),
            sampled_rows=int(cfg['retrieval'].get('graph_sampled_rows',1200)))
        model.propagation.install(weights,conf)
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


def losses(out,b,mean,std,cfg,mode,alignment_loss=None):
    valid=b['future_valid'].bool(); y=b['y_raw'].float();p=out['prediction'].float()*std+mean
    e=p[valid]-y[valid]
    if not e.numel():raise RuntimeError('No valid future labels in batch')
    main=e.abs().mean() if cfg['training'].get('forecast_loss','mae')=='mae' else F.smooth_l1_loss(p[valid],y[valid])
    rmse=torch.sqrt(e.square().mean()+1e-8)
    rank=main.new_zeros(());risk_loss=main.new_zeros(());prop_utility_loss=main.new_zeros(())
    if out['candidate_weights'] is not None and mode not in ('no_retrieval','no_future'):
        base=out['base_prediction'].detach().float()
        residual=out['candidate_residuals'].detach().float();yn=b['y_norm'].float()
        utility=(base[:,None]-yn[:,None]).abs()-(base[:,None]+residual-yn[:,None]).abs()
        available=out['candidate_available'].bool() & valid[:,None]
        margin=float(cfg['training'].get('utility_margin',0.02))
        positive=(utility>margin)&available
        negative=(utility<-margin)&available
        logits=out['candidate_logits'].float()
        relative=logits[:,:-1]-logits[:,-1:]
        sm=float(cfg['training'].get('score_margin',0.0))
        count=(positive.sum()+negative.sum()).clamp_min(1)
        rank=(F.softplus(sm-relative)*positive).sum()/count + (F.softplus(sm+relative)*negative).sum()/count
        proposal=base+out['ungated_correction'].detach().float()
        gain=(base-yn).abs()-(proposal-yn).abs()
        # Logit-space BCE is autocast-safe on V100 AMP and keeps the original sigmoid gate.
        risk_loss=F.binary_cross_entropy_with_logits(out['risk_logit'].float()[valid],(gain>margin).float()[valid])
        if out.get('candidate_expected_gain') is not None:
            predicted=out['candidate_expected_gain'].float()
            if available.any():
                prop_utility_loss=F.smooth_l1_loss(predicted[available],utility.detach()[available])
    base_error=(out['base_prediction'].float()*std+mean)[valid]-y[valid]
    base_loss=base_error.abs().mean()
    base_weight=(float(cfg['training'].get('base_preserve_weight',0.0)) if mode=='full' else 0.0)
    alignment=main.new_zeros(()) if alignment_loss is None else alignment_loss
    total=(main+float(cfg['training'].get('rmse_weight',0))*rmse
           +float(cfg['training'].get('rank_weight',0))*rank
           +float(cfg['training'].get('risk_weight',0))*risk_loss
           +float(cfg.get('propagation',{}).get('utility_weight',0.03))*prop_utility_loss
           +base_weight*base_loss
           +(float(cfg['training'].get('alignment_weight',0)) if mode=='full' else 0)*alignment)
    return total,{'mae':float(e.abs().mean().detach()),'rmse':float(rmse.detach()),
                  'ranking':float(rank.detach()),'risk':float(risk_loss.detach()),'propagation_utility':float(prop_utility_loss.detach()),'alignment':float(alignment.detach()),
                  'base_loss':float(base_loss.detach()),'gate':float(out['forecast_gate'].mean().detach()),
                  'loss':float(total.detach())}


def compute_alignment(model,b,repo,device,temperature=.1,out=None):
    """Train-only paired incomplete-query / complete-history InfoNCE.

    Pair identity is exact historical window_start, never current target future.
    If a training window lacks a complete-history counterpart, skip it.
    """
    if not hasattr(repo,'row_by_start'):
        repo.row_by_start={int(t):i for i,t in enumerate(repo.starts)}
    starts=b['window_start'].detach().cpu().tolist()
    paired=[(j,repo.row_by_start[int(w)]) for j,w in enumerate(starts) if int(w) in repo.row_by_start]
    if len(paired)<2:return torch.zeros((),device=device)
    rows=torch.tensor([p[0] for p in paired],device=device,dtype=torch.long)
    memory_ids=torch.tensor([p[1] for p in paired],device=device,dtype=torch.long)
    mx,mm,mt,md=repo.fetch(memory_ids,device)
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        memory_repr=model.memory_encoder(mx.float(),mm.float(),mt,md)
    if out is None or 'query_temporal_keys' not in out:
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            query=model.query_encoder(b['x'][rows].float(),b['mask'][rows].float(),
                                      b['tod'][rows].long(),b['dow'][rows].long())
        q={k:query[k] for k in ('temporal_key','spatial_key')}
    else:
        q={'temporal_key':out['query_temporal_keys'][rows],
           'spatial_key':out['query_spatial_keys'][rows]}
    t=float(temperature)
    loss=torch.zeros((),device=device)
    qstarts=b['window_start'][rows].long()
    overlap=(qstarts[:,None]-qstarts[None,:]).abs()<model.H+model.P
    # Do not treat near-identical or overlapping windows as negatives.
    for key in ('temporal_key','spatial_key'):
        a=F.normalize(q[key].float(),dim=-1);m=F.normalize(memory_repr[key].float().detach(),dim=-1)
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            logits=(a.float()@m.float().T)/max(t,1e-5)
        diag=torch.eye(len(rows),device=device,dtype=torch.bool)
        logits=logits.masked_fill(overlap & ~diag,-1e4)
        loss=loss+F.cross_entropy(logits,torch.arange(len(rows),device=device))
    return loss/2.

def _query_encoder_diagnostics(model, batch):
    """On-demand diagnostics; never silently replace non-finite evidence."""
    with torch.no_grad():
        faulty=[]
        for name, tensor in model.query_encoder.named_parameters():
            if not bool(torch.isfinite(tensor).all().item()):
                faulty.append(name)
        input_bad={name:int((~torch.isfinite(batch[name])).sum().item())
                   for name in ('x', 'mask') if name in batch}
        # Run the same query once with lightweight stage hooks ONLY after a
        # failure, so normal training has no extra GPU synchronizations.
        bad_modules=[]
        handles=[]
        def record(name):
            def hook(module, inputs, outputs):
                if torch.is_tensor(outputs) and not bool(torch.isfinite(outputs).all().item()):
                    bad_modules.append(name)
            return hook
        for name, module in model.query_encoder.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(record(name)))
        try:
            x=batch['x'].float()
            with torch.autocast(device_type=x.device.type,enabled=False):
                model.query_encoder(x,batch['mask'].float(),batch['tod'].long(),batch['dow'].long())
        except Exception as exc:
            bad_modules.append('diagnostic_forward_exception:'+str(exc)[:120])
        finally:
            for handle in handles:handle.remove()
        return {"nonfinite_parameters":faulty[:16],"first_nonfinite_linear_layers":bad_modules[:8],
                "nonfinite_inputs":input_bad}


def run_epoch(model,loader,index,repo,device,mean,std,cfg,mode,opt=None,ep=1,scaler=None):
    training=opt is not None;model.train(training);model.stla.base.eval();cum={};size=0
    skipped_updates=0
    mae_sum=0.;valid_sum=0.
    with torch.enable_grad() if training else torch.no_grad():
        for step,bb in enumerate(loader,1):
            b=move(bb,device)
            if training:opt.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=bool(cfg['runtime'].get('amp',False)) and device.type=='cuda'):
                    out=model(b,index,repo,mode=mode)
                    align=(compute_alignment(model,b,repo,device,float(cfg['training'].get('alignment_temperature',.1)),out=out)
                           if training and mode=='full' and float(cfg['training'].get('alignment_weight',0))>0 else None)
                    loss,parts=losses(out,b,mean,std,cfg,mode,align)
                    with torch.no_grad():
                        check_mask=b['future_valid'].bool()
                        mae_sum+=float((((out['prediction'].float()*std+mean)-b['y_raw'].float()).abs()*check_mask).sum())
                        valid_sum+=float(check_mask.sum())
            except RuntimeError as exc:
                if 'Non-finite' in str(exc) or 'non-finite' in str(exc):
                    print(f'QUERY_FAILURE_DIAGNOSTICS epoch={ep} batch={step} '
                          f'{_query_encoder_diagnostics(model,b)}',flush=True)
                raise
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f'Non-finite training loss at epoch={ep} batch={step} '
                                   f'mode={mode}; inspect encoder and retrieval health logs')
            if training:
                trainable=[p for p in model.parameters() if p.requires_grad]
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    norm=nn.utils.clip_grad_norm_(trainable,float(cfg['training']['grad_clip']))
                    if not bool(torch.isfinite(norm).item()):
                        # Do NOT modify parameters or EMA after an invalid backward.
                        # Dynamic loss scale is reduced before retrying next batch.
                        skipped_updates+=1
                        print(f'GRADIENT_OVERFLOW epoch={ep} batch={step} '
                              f'scale={scaler.get_scale():.1f} skipped={skipped_updates}',flush=True)
                        opt.zero_grad(set_to_none=True)
                        scaler.update(new_scale=max(1.0,float(scaler.get_scale())/2.0))
                        if skipped_updates>=20:
                            raise RuntimeError('Repeated non-finite gradients; '
                                               'stopping before corrupting Query Encoder')
                    else:
                        skipped_updates=0
                        old_scale=scaler.get_scale()
                        scaler.step(opt)
                        scaler.update()
                        if mode=='full' and scaler.get_scale()>=old_scale:
                            model.momentum_update(float(cfg['training'].get('memory_momentum',0.995)))
                else:
                    loss.backward()
                    norm=nn.utils.clip_grad_norm_(trainable,float(cfg['training']['grad_clip']))
                    if not bool(torch.isfinite(norm).item()):
                        raise RuntimeError(f'Non-finite gradients at epoch={ep} batch={step}')
                    opt.step()
                    if mode=='full':model.momentum_update(float(cfg['training'].get('memory_momentum',0.995)))
            bs=int(b['x'].shape[0]);size+=bs
            for k,v in parts.items():cum[k]=cum.get(k,0)+v*bs
            if training and step%int(cfg['runtime']['print_every'])==0:
                print(f'epoch={ep:03d} batch={step:05d}/{len(loader):05d} MAE={cum["mae"]/size:.5f} rank={cum["ranking"]/size:.5f}',flush=True)
    result={k:v/max(1,size) for k,v in cum.items()}
    result['mae']=mae_sum/max(valid_sum,1.)
    return result


def predict(model,loader,index,repo,device,mean,std,mode='full',amp=False):
    arrays={k:[] for k in ('prediction','base','true','valid','direct','correction')};g=[];model.eval();cand_good=0.;cand_total=0.;weighted_gain=0.
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
            if o.get('candidate_residuals') is not None:
                with torch.no_grad():
                    r=o['candidate_residuals'].float()
                    base=o['base_prediction'].detach()
                    util=(base[:,None]-b['y_norm'][:,None]).abs()-(base[:,None]+r-b['y_norm'][:,None]).abs()
                    usable=o['candidate_available'].bool()
                    v=b['future_valid'].bool()
                    cand_good+=float((((util>0)&usable).any(1)&v).sum())
                    cand_total+=float(v.sum())
                    weighted_gain+=float(((o['candidate_weights'][:,:r.shape[1]]*util).sum(1)*v).sum())
    result={k:np.concatenate(v) for k,v in arrays.items()};result['gate_mean']=float(np.mean(g))
    result['useful_candidate_recall']=cand_good/max(1.,cand_total)
    result['weighted_candidate_utility_normalized']=weighted_gain/max(1.,cand_total)
    return result


def train(cfg,cache,checkpoint,run_dir,smoke=False,resume=None):
    seed_all(int(cfg['experiment']['seed']));dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,ck=build_model(cfg,checkpoint,dev)
    graph_path=setup_graph(model,cfg,cache)
    mean=float(ck['mean']);std=float(ck['std']);limit=24 if smoke else None
    tr=SupportSplit(cache,'train',limit);va=SupportSplit(cache,'val',limit);repo=FullTrainRepository(cache,256 if smoke else None)
    model.set_future_reference(train_future_reference(repo,model.P,model.query_encoder.N))
    print('TRAIN_ONLY_FUTURE_REFERENCE_PASS shape='+str(tuple(model.future_reference.shape)),flush=True)
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
        if mode!='no_retrieval' and (index is None or ((ep-1)%interval==0)):
            t0=time.time();model.sync_region_projector();index=build_memory_index(model.memory_encoder,repo,dev,
                                                      16 if smoke else int(cfg['retrieval']['store_batch_size']),
                       propagation=model.propagation if model.propagation_enabled else None);index.to(dev)
            print(f'MEMORY_REFRESH epoch={ep} time={time.time()-t0:.1f}s',flush=True)
        if dev.type=='cuda':torch.cuda.reset_peak_memory_stats(dev)
        t0=time.time();a=run_epoch(model,tl,index,repo,dev,mean,std,cfg,mode,opt,ep,scaler)
        train_seconds=time.time()-t0
        model.sync_region_projector();val_index=(build_memory_index(model.memory_encoder,repo,dev,
                   16 if smoke else int(cfg['retrieval']['store_batch_size']),propagation=model.propagation if model.propagation_enabled else None) if mode!='no_retrieval' else None)
        if val_index is not None:val_index.to(dev)
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
        persist_epoch(run_dir,model,opt,scaler,cfg,checkpoint,ck,'PCFT-STLA',ep,best,best_ep,history,stale,ph,improved)
        if improved and val_index is not None:
            val_index.save(run_dir/'best_memory_index.npz',{
                'version':cfg['version'],'memory_rows':len(repo),'dataset':cfg['experiment']['dataset'],
                'target':cfg['experiment']['target'],'graph':str(graph_path)})
        if stale>=int(cfg['training']['patience']) and (ph=='full' or cfg['training'].get('schedule')=='frozen'):
            print(f'EARLY_STOP epoch={ep} best_epoch={best_ep}',flush=True);break
    best_path=run_dir/'best.pt'
    if not best_path.is_file():raise RuntimeError('No completed best checkpoint')
    saved=torch.load(best_path,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True);model.to(dev)
    if mode!='no_retrieval':
        model.sync_region_projector();index=build_memory_index(model.memory_encoder,repo,dev,16 if smoke else int(cfg['retrieval']['store_batch_size']),propagation=model.propagation if model.propagation_enabled else None)
        index.to(dev);index.save(run_dir/'best_memory_index.npz',{'version':cfg['version'],'memory_rows':len(repo),
            'dataset':cfg['experiment']['dataset'],'target':cfg['experiment']['target'],'graph':str(graph_path)})
    else:index=None
    pred=predict(model,vl,index,repo,dev,mean,std,mode,amp=scaler.is_enabled())
    vm=metrics(pred['prediction'],pred['true'],pred['valid'],pred['direct'])
    bm=metrics(pred['base'],pred['true'],pred['valid'],pred['direct'])
    np.savez_compressed(run_dir/'val_predictions.npz',**{k:v.astype(np.float32) for k,v in pred.items() if isinstance(v,np.ndarray)})
    save_summary(run_dir/'summary.json',{'model':'PCFT-STLA-v1.0.0','best_epoch':best_ep,
        'validation':vm,'base_validation':bm,'gate_mean':pred['gate_mean'],
        'backbone_checkpoint':str(checkpoint),'ablation':mode,'test_evaluated':False})
    print(f'TRAIN_PASS run_dir={run_dir} best_epoch={best_ep} val_MAE={vm["MAE"]:.6f}',flush=True)
    return run_dir/'best.pt'
