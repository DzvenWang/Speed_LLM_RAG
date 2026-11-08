"""Complete v2 RAFT training, validation and evidence attribution."""
from __future__ import annotations
import copy, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .v3.losses import mechanism_losses
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from .data import SupportSplit, RetrievalCache
from .evidence import FutureMemory, shortlist, counterfactual_residuals
from .stla import load_pretrained_stla, MaskAwareSTLAformer
from .model import TCRRSTLA
from .evaluation import metrics, save_summary
from .config import file_sha256
from .checkpoint import persist_epoch,restore_training
from .replay_cache import prepare_bank
from .normal_teacher import prepare_normal_teacher_bank
from .wrong_evidence import legal_wrong_indices


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True;torch.backends.cudnn.benchmark=False


def phase(ep,cfg):
    if cfg['training'].get('schedule','progressive')=='frozen':return 'warmup'
    if ep<=int(cfg['training']['warmup_epochs']):return 'warmup'
    if ep<=int(cfg['training']['partial_epochs']):return 'partial'
    return 'full'


def build_model(cfg,checkpoint,device):
    base,ck=load_pretrained_stla(Path(cfg['project']['root']),Path(checkpoint),device)
    m=TCRRSTLA(MaskAwareSTLAformer(base),cfg).to(device)
    if cfg.get('traffic',{}).get('enabled',True):
        from .traffic_neighbors import train_only_neighbours
        formal=Path(cfg['project']['root'])/'Base_SupportBench'  # resolved via protocol cache below
        from .config import cache_path
        train_cache=cache_path(cfg,'formal')
        if (train_cache/'train_x_full_norm.npy').is_file():
            nbr=train_only_neighbours(train_cache,int(cfg.get('traffic',{}).get('graph_neighbors',8)))
            m.traffic.install_neighbors(nbr.to(device))
            print(f'TCRR_TRAIN_NEIGHBOR_GRAPH nodes={nbr.shape[0]} degree={nbr.shape[1]}',flush=True)
    return m,ck


def optimizer(model,cfg):
    bp=list(model.stla.base.parameters()); ids={id(p) for p in bp}
    new=[p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{'params':new,'lr':float(cfg['training']['lr_new'])},
                              {'params':bp,'lr':float(cfg['training']['lr_backbone'])}],
                             weight_decay=float(cfg['training']['weight_decay']))


def _batch_to_device(batch,device):
    return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}


def _make_loader(ds,batch_size,shuffle,cfg,smoke=False):
    nw=0 if smoke else int(cfg['runtime'].get('num_workers',0))
    kw=dict(batch_size=int(batch_size),shuffle=shuffle,num_workers=nw,
            pin_memory=bool(cfg['runtime'].get('pin_memory',True)) and torch.cuda.is_available())
    if nw:kw.update(persistent_workers=bool(cfg['runtime'].get('persistent_workers',True)),
                    prefetch_factor=int(cfg['runtime'].get('prefetch_factor',2)))
    return DataLoader(ds,**kw)


def evidence_for_batch(model,b,cache,memory,device,cfg,mode,bank=None,teacher=None,normal_bank=None):
    if mode=='no_retrieval':return {}
    indices=b['index'].detach().cpu().numpy()
    ids,scores,cov=shortlist(cache,indices,int(cfg['retrieval']['shortlist_per_period']),device)
    if mode=='cross_history':
        import pandas as pd
        # Deterministic across batch sizes; no Query future information used.
        from pathlib import Path
        starts=pd.read_csv(Path(memory.arrays['x'].filename).parent/'train_manifest.csv').window_start.to_numpy(dtype=np.int64)
        wrong=legal_wrong_indices(b['window_start'],ids.shape[1],starts[:len(memory)],
                    int(cfg['experiment']['history'])+int(cfg['experiment']['horizon']),device)
        ids=wrong
    candidates=memory.fetch(ids,device)
    query_mask=b['mask'].float();query_x=b['x'].float()
    # Only features available in the actual query participate in reranking.
    dist=((candidates['x']-query_x[:,None]).abs()*query_mask[:,None]).sum(2) / query_mask[:,None].sum(2).clamp_min(1.)
    replay='full' if mode=='no_mask_replay' else 'matched'
    if bank is not None and replay=='matched' and mode!='cross_history':
        residual=bank.fetch(b['index'],device)
    else:
        residual=counterfactual_residuals(teacher if teacher is not None else model.stla,
                  candidates,query_mask,int(cfg['retrieval']['replay_batch_size']),replay,
                  strict_source_mask=str(cfg.get('version','')).startswith(('4.','5.')))
    normal=normal_bank.fetch(ids,device) if normal_bank is not None else None
    with torch.no_grad():
        teacher_base=(teacher(query_x,query_mask,b['tod'].long(),b['dow'].long())[0]
                      if teacher is not None and not str(cfg.get('version','')).startswith(('4.','5.')) else None)
    return dict(candidate_ids=ids,candidate_histories=candidates['x'],candidate_source_valid=candidates.get('source_valid'),normal_residual=normal,teacher_base=teacher_base,candidate_y=candidates['y'],candidate_valid=candidates['future_valid'],residual=residual,
                candidate_scores=scores,candidate_coverage=cov,hist_distance=dist)


def compute_losses(out,b,mean,std,cfg,mode):
    valid=b['future_valid'].bool(); raw=out['prediction'].float()*std+mean
    y=b['y_raw'].float();kind=str(cfg['training'].get('forecast_loss','mae'))
    err=raw[valid]-y[valid]
    if err.numel()==0:raise RuntimeError('No valid future labels in training batch')
    main=err.abs().mean() if kind=='mae' else F.smooth_l1_loss(raw[valid],y[valid])
    risk=torch.sqrt((err.square()).mean()+1e-8)
    rank=main.new_zeros(())
    risk_loss=main.new_zeros(())
    utility_loss=main.new_zeros(())
    if out['candidate_weights'] is not None and mode not in ('no_retrieval','no_evidence'):
        residual=out['candidate_residuals'].float().detach()
        base=out['base_prediction'].float().detach()
        yt=b['y_norm'].float()
        utility=(base[:,None]-yt[:,None]).abs()-(base[:,None]+residual-yt[:,None]).abs()
        avail=out['candidate_available'].bool()&valid[:,None]
        margin=float(cfg['training'].get('utility_margin',0.02))
        positive=(utility>margin)&avail
        negative=(utility<-margin)&avail
        logits=out['candidate_logits'].float()
        relative=logits[:,:-1]-logits[:,-1:]
        score_margin=float(cfg['training'].get('score_margin',0.0))
        p=F.softplus(score_margin-relative)
        n=F.softplus(score_margin+relative)
        count=(positive.sum()+negative.sum()).clamp_min(1)
        rank=((p*positive).sum()+(n*negative).sum())/count
        # A paired, forecast-space risk target checks *combined* evidence,
        # rather than treating individually good candidates as additive.
        ungated=out['ungated_correction'].detach().float()
        proposed=base+ungated
        gain=(base-yt).abs()-(proposed-yt).abs()
        target=(gain>margin).float()
        # Logit-space BCE is autocast-safe on V100 AMP and keeps the original sigmoid gate.
        risk_loss=F.binary_cross_entropy_with_logits(out['risk_logit'].float()[valid],target[valid])
        if out.get('candidate_expected_gain') is not None and mode not in ('no_candidate_utility','no_utility'):
            # Candidate label matches the actual bounded evidence amplitude.
            delta=out['candidate_residuals'].detach().float()
            limit=float(cfg['model']['max_normalized_correction'])
            proposal_delta=limit*torch.tanh(delta/limit)
            gain_candidate=(base[:,None]-yt[:,None]).abs()-(base[:,None]+proposal_delta-yt[:,None]).abs()
            pred_candidate=out['candidate_expected_gain'].float()
            utility_loss=F.smooth_l1_loss(pred_candidate[avail],gain_candidate.detach()[avail]) if avail.any() else utility_loss
            if out.get('candidate_expected_harm') is not None and avail.any():
                harm_candidate=(-gain_candidate).clamp_min(0.)
                utility_loss=utility_loss+0.25*F.smooth_l1_loss(pred_candidate.new_tensor(0.)+out['candidate_expected_harm'].float()[avail],harm_candidate.detach()[avail])
            if out.get('mixture_expected_gain') is not None and mode not in ('no_mixture_utility','no_utility'):
                mixture_pred=out['mixture_expected_gain'].float()[valid]
                utility_loss=utility_loss+F.smooth_l1_loss(mixture_pred,gain.detach()[valid])
                if out.get('mixture_expected_harm') is not None:
                    harm=(-gain).clamp_min(0.)
                    utility_loss=utility_loss+0.25*F.smooth_l1_loss(F.softplus(out['mixture_expected_harm'].float()[valid]),harm.detach()[valid])
    loss=main+float(cfg['training'].get('rmse_weight',0.0))*risk+float(cfg['training'].get('rank_weight',0.0))*rank+float(cfg['training'].get('risk_weight',0.0))*risk_loss+float(cfg['training'].get('traffic_utility_weight',0.03))*utility_loss
    if str(cfg.get('version','')).startswith('3.') and mode!='no_retrieval':
        cfg3=cfg.get('v3',{})
        extra,extra_stats=mechanism_losses(out,b,cf_target=b.get('cf_target_training_only'),
            cf_weight=float(cfg3.get('cf_weight',.02)),
            ranking_weight=float(cfg3.get('ranking_weight',.03)),
            harm_weight=float(cfg3.get('harm_weight',.02)),
            control_weight=float(cfg3.get('control_weight',.01)))
        loss=main+float(cfg['training'].get('rmse_weight',0.05))*risk+extra
    return loss,dict(mae=float(err.abs().mean().detach()),rmse=float(risk.detach()),ranking=float(rank.detach()),risk=float(risk_loss.detach()),utility=float(utility_loss.detach()),
                     gate=float(out['gate'].mean().detach()),loss=float(loss.detach()))


def run_epoch(model,loader,rc,memory,device,mean,std,cfg,mode,opt=None,epoch=1,bank=None,teacher=None,scaler=None,normal_bank=None):
    tr=opt is not None;model.train(tr); model.stla.base.eval(); total={}; seen=0
    mae_sum=0.;valid_sum=0.
    context=torch.enable_grad() if tr else torch.no_grad()
    with context:
        for bi,raw_batch in enumerate(loader,1):
            batch=_batch_to_device(raw_batch,device)
            if tr:opt.zero_grad(set_to_none=True)
            ev=evidence_for_batch(model,batch,rc,memory,device,cfg,mode,bank=bank,teacher=teacher,normal_bank=normal_bank)
            if str(cfg.get('version','')).startswith('3.') and mode!='no_retrieval' and 'x_full_training_only' in batch:
                if teacher is None:raise RuntimeError('v3 requires one frozen paired teacher')
                with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
                    ref_full=teacher(batch['x_full_training_only'].float(),torch.ones_like(batch['mask']).float(),
                                     batch['tod'].long(),batch['dow'].long())[0].float()
                    if ev.get('teacher_base') is None:raise RuntimeError('Missing masked teacher response')
                    batch['cf_target_training_only']=ref_full-ev['teacher_base'].float().detach()
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=bool(cfg['runtime'].get('amp',False)) and device.type=='cuda'):
                out=model(batch['x'].float(),batch['mask'].float(),batch['tod'].long(),batch['dow'].long(),mode=mode,**ev)
                loss,ls=compute_losses(out,batch,mean,std,cfg,mode)
                with torch.no_grad():
                    check_mask=batch['future_valid'].bool()
                    mae_sum+=float((((out['prediction'].float()*std+mean)-batch['y_raw'].float()).abs()*check_mask).sum())
                    valid_sum+=float(check_mask.sum())
            if tr:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward();scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training']['grad_clip']))
                    scaler.step(opt);scaler.update()
                else:
                    loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training']['grad_clip']));opt.step()
            bs=int(batch['x'].shape[0]);seen+=bs
            for k,v in ls.items():total[k]=total.get(k,0.)+v*bs
            if tr and bi%int(cfg['runtime']['print_every'])==0:
                print(f'epoch={epoch:03d} batch={bi:05d}/{len(loader):05d} MAE={total["mae"]/seen:.5f} rank={total["ranking"]/seen:.5f}',flush=True)
    result={k:v/max(1,seen) for k,v in total.items()}
    result['mae']=mae_sum/max(valid_sum,1.)
    return result


def predict(model,loader,rc,memory,device,mean,std,cfg,mode='full',bank=None,teacher=None,normal_bank=None):
    all_data={k:[] for k in ('prediction','base','true','valid','direct','correction')}; gates=[];harm_probs=[];cand_good=0.;cand_total=0.;weighted_gain=0.;oracle_gain=0.;realized_gain=0.;harm_count=0.;corrected_count=0.
    model.eval()
    with torch.no_grad():
        for bb in loader:
            b=_batch_to_device(bb,device);ev=evidence_for_batch(model,b,rc,memory,device,cfg,mode,bank=bank,teacher=teacher,normal_bank=normal_bank)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=bool(cfg['runtime'].get('amp',False)) and device.type=='cuda'):
                o=model(b['x'].float(),b['mask'].float(),b['tod'].long(),b['dow'].long(),mode=mode,**ev)
            for k,v in [('prediction',o['prediction']),('base',o['base_prediction'])]:
                all_data[k].append((v.float()*std+mean).cpu().numpy())
            all_data['true'].append(b['y_raw'].cpu().numpy()); all_data['valid'].append(b['future_valid'].cpu().numpy())
            all_data['direct'].append(b['direct_mask'].cpu().numpy())
            all_data['correction'].append((o['correction'].float()*std).cpu().numpy())
            gates.append(float(o['gate'].mean()))
            with torch.no_grad():
                valid=b['future_valid'].bool()
                gain_act=(o['base_prediction'].detach().float()-b['y_norm']).abs()-(o['prediction'].detach().float()-b['y_norm']).abs()
                realized_gain+=float((gain_act*valid).sum())
                harm_count+=float(((gain_act<0)&valid).sum())
                corrected_count+=float((((o['prediction']-o['base_prediction']).abs()>1e-4)&valid).sum())
            if o.get('harm_probability') is not None:
                harm_probs.append(o['harm_probability'].float().detach().cpu().numpy())
            if o.get('candidate_residuals') is not None:
                with torch.no_grad():
                    r=o['candidate_residuals'].float()
                    if str(cfg.get('version','')).startswith('4.'):
                        bound=float(cfg['model']['max_normalized_correction'])
                        actual=bound*torch.tanh(r/bound)
                    else:actual=r
                    residual_gain=(o['base_prediction'].detach()[:,None]-b['y_norm'][:,None]).abs()-(o['base_prediction'].detach()[:,None]+actual-b['y_norm'][:,None]).abs()
                    available=o['candidate_available'].bool()
                    good=((residual_gain>0)&available).any(1)
                    v=b['future_valid'].bool()
                    cand_good+=float((good&v).sum());cand_total+=float(v.sum())
                    w=o['candidate_weights'][:,:r.shape[1]].float()
                    weighted_gain+=float(((w*residual_gain).sum(1)*v).sum())
                    best=residual_gain.masked_fill(~available,-1e4).max(1).values.clamp_min(0)
                    oracle_gain+=float((best*v).sum())
    out={k:np.concatenate(v) for k,v in all_data.items()};out['gate_mean']=float(np.mean(gates))
    out['useful_candidate_recall']=cand_good/max(1.,cand_total)
    out['weighted_candidate_utility_normalized']=weighted_gain/max(1.,cand_total)
    out['oracle_candidate_gain_normalized']=oracle_gain/max(1.,cand_total)
    observed_valid=float(out['valid'].astype(bool).sum())
    out['realized_gain_normalized']=realized_gain/max(1.,observed_valid)
    out['harm_rate']=harm_count/max(1.,observed_valid)
    out['correction_coverage']=corrected_count/max(1.,observed_valid)
    if harm_probs:out['harm_probability']=np.concatenate(harm_probs,axis=0)
    return out


def train(cfg,cache,ckpt,retrieval_train,retrieval_val,run_dir,smoke=False,resume=None):
    seed_all(int(cfg['experiment']['seed']));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,bc=build_model(cfg,ckpt,device)
    mean=float(bc['mean']);std=float(bc['std']);limit=24 if smoke else None
    cf_source=str(cfg.get('version','')).startswith('3.') and cfg.get('v3',{}).get('contribution','M3')!='M0'
    tr=SupportSplit(cache,'train',limit,cf_training=cf_source)
    va=SupportSplit(cache,'val',limit,cf_training=False)
    memory=FutureMemory(cache,256 if smoke else None)
    if not smoke and cfg['runtime'].get('preload_retrieval_memory_gpu',False):
        loaded=memory.preload(device);print(f'RETRIEVAL_MEMORY_GPU loaded={loaded}',flush=True)
    mode=str(cfg['training'].get('ablation','full'))
    if str(cfg.get('version','')).startswith('3.') and not cfg['runtime'].get('frozen_replay_teacher',True):
        raise RuntimeError('TCRR v3 CF attribution requires a frozen replay teacher')
    rt=RetrievalCache(retrieval_train) if retrieval_train else None;rv=RetrievalCache(retrieval_val) if retrieval_val else None
    bs=3 if smoke else int(cfg['training']['batch_size'])
    tl=_make_loader(tr,bs,True,cfg,smoke);vl=_make_loader(va,bs,False,cfg,smoke)
    # A frozen teacher is a deliberate v2.1 algorithm setting. Its residual
    # evidence is computed once and reused while the query STLA may fine-tune.
    teacher=copy.deepcopy(model.stla).to(device).eval() if cfg['runtime'].get('frozen_replay_teacher',True) else None
    if teacher is not None:
        for p in teacher.parameters():p.requires_grad_(False)
    normal_bank=(prepare_normal_teacher_bank(cfg,memory,teacher,ckpt,device,smoke=smoke)
                 if teacher is not None and mode not in ('no_retrieval',) else None)
    banktrain=bankval=None
    if teacher is not None and mode not in ('no_retrieval','no_mask_replay'):
        banktrain=prepare_bank(cfg,'train',tr,rt,memory,teacher,ckpt,retrieval_train,device,smoke=smoke)
        bankval=prepare_bank(cfg,'val',va,rv,memory,teacher,ckpt,retrieval_val,device,smoke=smoke)
    opt=optimizer(model,cfg)
    scaler=torch.amp.GradScaler('cuda',enabled=bool(cfg['runtime'].get('amp',False)) and device.type=='cuda')
    best=float('inf');best_epoch=0;stale=0;history=[];prev_phase=None;start_epoch=1
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    if resume:
        resumed=restore_training(resume,model,opt,scaler,cfg,ckpt)
        best=float(resumed['best_val_mae']);best_epoch=int(resumed['best_epoch']);
        stale=int(resumed['stale']);history=list(resumed['history'])
        prev_phase=resumed['previous_phase'];start_epoch=int(resumed['epoch'])+1
    max_epoch=2 if smoke else int(cfg['training']['max_epochs']);patience=int(cfg['training']['patience'])
    print(f'PERF_REPLAY_MODE teacher={"frozen_cached" if teacher is not None else "dynamic"} batch={bs} amp={scaler.is_enabled()}',flush=True)
    for ep in range(start_epoch,max_epoch+1):
        ph=phase(ep,cfg);model.set_train_phase(ph,int(cfg['training']['partial_last_layers']))
        if ph!=prev_phase:
            stale=0;prev_phase=ph;print(f'TRAIN_PHASE epoch={ep} phase={ph}',flush=True)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        t0=time.time();a=run_epoch(model,tl,rt,memory,device,mean,std,cfg,mode,opt,ep,banktrain,teacher,scaler,normal_bank)
        train_seconds=time.time()-t0
        v=run_epoch(model,vl,rv,memory,device,mean,std,cfg,mode,None,ep,bankval,teacher,scaler,normal_bank)
        val_seconds=time.time()-t0-train_seconds
        rec={'epoch':ep,'phase':ph,**{f'train_{k}':x for k,x in a.items()},**{f'val_{k}':x for k,x in v.items()},
             'seconds':time.time()-t0,'train_seconds':train_seconds,'val_seconds':val_seconds,
             'train_samples_per_second':len(tr)/max(train_seconds,1e-5),
             'peak_gpu_gib':torch.cuda.max_memory_reserved(device)/1024**3 if device.type=='cuda' else 0.0}
        history.append(rec);print('EPOCH '+json.dumps(rec),flush=True)
        improved=v['mae'] < best-float(cfg['training']['min_delta'])
        if improved:best=v['mae'];best_epoch=ep;stale=0
        else:stale+=1
        pd.DataFrame(history).to_csv(run_dir/'history.csv',index=False)
        persist_epoch(run_dir,model,opt,scaler,cfg,ckpt,bc,'TCRR-STLA',ep,best,best_epoch,history,stale,prev_phase,improved)
        if stale>=patience and (ph=='full' or cfg['training'].get('schedule')=='frozen'):
            print(f'EARLY_STOP epoch={ep} best_epoch={best_epoch}',flush=True);break
    best_path=run_dir/'best.pt'
    if not best_path.is_file():raise RuntimeError('No completed best.pt checkpoint')
    saved=torch.load(best_path,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True);model.to(device)
    p=predict(model,vl,rv,memory,device,mean,std,cfg,mode,bankval,teacher,normal_bank)
    vm=metrics(p['prediction'],p['true'],p['valid'],p['direct']);bm=metrics(p['base'],p['true'],p['valid'],p['direct'])
    np.savez_compressed(run_dir/'val_predictions.npz',**{k:v.astype(np.float32) for k,v in p.items() if isinstance(v,np.ndarray)})
    save_summary(run_dir/'summary.json',{'model':f'TCRR-STLA-v{cfg["version"]}','best_epoch':best_epoch,
                 'validation':vm,'base_validation':bm,'gate_mean':p['gate_mean'],
                 'backbone_checkpoint':str(ckpt),'ablation':mode,
                 'replay_mode':'frozen_cached' if teacher is not None else 'dynamic','test_evaluated':False})
    print(f'TRAIN_PASS run_dir={run_dir} best_epoch={best_epoch} val_MAE={vm["MAE"]:.6f}',flush=True)
    return run_dir/'best.pt'
