from __future__ import annotations
import copy, csv, json, random, time, os, resource, datetime
from pathlib import Path
import numpy as np,torch
from torch import nn
from .config import json_write,sha256
from .data import loader,to_device
from .losses import losses, forecast_validation_loss
from .evaluation import evaluate
from .online_diagnostics import OnlineDiagnostics, append_diagnostics

PHASE_ORDER=['retrieval','recovery','injection','partial','full']

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def save_ckpt(path,model,opt,cfg,meta,epoch,phase,validation,memory_path=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={'version':str(cfg.get('version','2.0.0')),'model':'FARE','phase':phase,'epoch':epoch,'state_dict':model.state_dict(),'optimizer':opt.state_dict() if opt is not None else None,'config':cfg,'metadata':meta,'validation':validation,'memory_path':str(memory_path) if memory_path else None,'rng':{'torch':torch.get_rng_state(),'numpy':np.random.get_state()}}
    torch.save(state,path)

def load_ckpt(path,model=None,device='cpu',restore_rng=False):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['model']!='FARE':raise ValueError('Not a FARE checkpoint')
    if model is not None:
        model.load_state_dict(ck['state_dict'],strict=True);model.to(device)
    if restore_rng:
        torch.set_rng_state(ck['rng']['torch']);np.random.set_state(ck['rng']['numpy'])
    return ck

class FARETrainer:
    def __init__(self,model,memory,train_ds,val_ds,cfg,meta,device,run_dir,memory_path=None):
        self.model=model.to(device);self.memory=memory.to(device);self.train_ds=train_ds;self.val_ds=val_ds
        self.cfg=cfg;self.meta=meta;self.device=device;self.run_dir=Path(run_dir);self.run_dir.mkdir(parents=True,exist_ok=True)
        self.memory_path=memory_path;self.mean=float(meta['mean']);self.std=float(meta['std']);self.global_best=float('inf')
        self.history=[];self.phase_history=[]
        self.diag_config=cfg.get('diagnostics',{})
        self._last_epoch_diag={}
        self.progress_every=int(self.diag_config.get('progress_every_batches',500))
        if self.progress_every < 0:raise ValueError('progress_every_batches must be >= 0')
        self.resource_log_prefix=os.environ.get('FARE_RESOURCE_OUTPUT_PREFIX')
        # Reuse DataLoader instances, allowing persistent_workers across epochs.
        bs=self.cfg['training']['batch_size'];nw=self.cfg['training'].get('num_workers',0)
        self._train_loader=loader(self.train_ds,bs,shuffle=True,workers=nw)
        self._val_loader=loader(self.val_ds,bs,shuffle=False,workers=nw)
        json_write(self.run_dir/'run_manifest.json',{'cfg':cfg,'meta':meta,'memory_path':str(memory_path),'start_time':time.strftime('%Y-%m-%d %H:%M:%S')})
    def _optimizer(self,phase):
        base=[];new=[]
        for name,p in self.model.named_parameters():
            if not p.requires_grad:continue
            (base if name.startswith('stla.base.') else new).append(p)
        if not new and not base:raise RuntimeError('No trainable parameters')
        lr=self.cfg['training'];groups=[]
        new_lr=(lr.get('finetune_new_lr',lr['new_lr']) if phase in ('partial','full') else lr['new_lr'])
        if new:groups.append({'params':new,'lr':new_lr})
        if base:groups.append({'params':base,'lr':lr['partial_backbone_lr'] if phase=='partial' else lr['full_backbone_lr']})
        return torch.optim.AdamW(groups,weight_decay=lr['weight_decay'])
    def _epoch(self,phase,optimizer=None,max_batches=None):
        training=optimizer is not None
        self.model.train(training)
        if not training:self.model.eval()
        it=self._train_loader if training else self._val_loader
        acc={};total=0
        epoch_started=time.monotonic()
        diag=(OnlineDiagnostics(every_batches=self.diag_config.get('sample_every_batches',50),
                                max_batches=self.diag_config.get('max_sampled_batches',160))
              if self.diag_config.get('enabled',True) else None)
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
        mode=self.cfg['experiment'].get('training_mode','full')
        opt_cfg=self.cfg.get('performance',{})
        # These skip only computations whose outputs cannot influence the
        # phase objective/selection score. Full inference/phase evaluation stays intact.
        retrieval_fast=(phase=='retrieval' and mode!='no_memory' and
                        opt_cfg.get('retrieval_phase_fast',True))
        forecast_only_val=(not training and phase in ('injection','partial','full') and
                           opt_cfg.get('forecast_only_validation',True))
        debug=(opt_cfg.get('strict_batch_diagnostics',False) or
               __import__('os').environ.get('FARE_DEBUG_NONFINITE')=='1')
        params=[p for p in self.model.parameters() if p.requires_grad] if training else []
        for j,batch in enumerate(it):
            if max_batches is not None and j>=max_batches:break
            bt=to_device(batch,self.device)
            if training:optimizer.zero_grad(set_to_none=True)
            grad_norm=None
            with torch.set_grad_enabled(training):
                out=self.model(bt,self.memory,mode=mode,
                               phase_fast='retrieval' if retrieval_fast else None)
                if debug:
                    for key,value in out.items():
                        if isinstance(value,torch.Tensor) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                            raise FloatingPointError(f'FARE_NONFINITE_FORWARD phase={phase} batch={j} key={key}')
                if forecast_only_val:
                    ls=forecast_validation_loss(bt,out,self.mean,self.std)
                else:
                    ls=losses(self.model,bt,out,self.memory,phase,self.mean,self.std,self.cfg)
                if not bool(torch.isfinite(ls['total'])):
                    raise FloatingPointError(f'FARE_NONFINITE_LOSS phase={phase} batch={j} key=total')
                if training:
                    ls['total'].backward()
                    try:
                        grad_norm=nn.utils.clip_grad_norm_(params,self.cfg['training']['grad_clip'],
                                                 error_if_nonfinite=True,foreach=True)
                    except RuntimeError as exc:
                        # Only on failure, find the offending parameter.
                        bad=[]
                        for name,parameter in self.model.named_parameters():
                            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                                bad.append(name)
                        raise FloatingPointError(
                            f'FARE_NONFINITE_GRAD phase={phase} batch={j} parameters={bad[:20]}') from exc
                    optimizer.step()
            bs=len(bt['x_obs']);total+=bs
            if diag is not None and diag.should_sample(j):
                diag.update(bt,out,self.mean,self.std,grad_norm=grad_norm)
            if self.progress_every and (j+1)%self.progress_every==0:
                # No CUDA sync: the displayed rate is a progress heartbeat,
                # not a precise CUDA kernel throughput benchmark.
                print('FARE_PROGRESS '+json.dumps({
                    'phase':phase,'training':training,'batch':j+1,'examples':total,
                    'elapsed_wall_seconds':round(time.monotonic()-epoch_started,1)
                }),flush=True)
            # Accumulate detached tensors on GPU; synchronize once at epoch end,
            # rather than for every metric at every batch.
            def add_stat(key,value):
                term=value.detach()*bs
                acc[key]=acc[key]+term if key in acc else term
            if 'retrieval_valid' in out:
                add_stat('retrieval_valid_groups',out['retrieval_valid'].float().mean())
                if 'candidate_prediction' in out:
                    add_stat('candidate_delta',(out['candidate_prediction']-out['base_prediction']).abs().mean())
                if 'final_gate' in out:
                    add_stat('final_gate_mean',out['final_gate'].mean())
            for key,value in ls.items():
                add_stat(key,value)
        self._last_epoch_diag=diag.summary() if diag is not None else {}
        self._last_epoch_diag['examples_processed']=total
        self._last_epoch_diag['wall_seconds_unsynchronized']=round(time.monotonic()-epoch_started,2)
        if self.device.type=='cuda':
            self._last_epoch_diag['cuda_peak_allocated_gb']=round(torch.cuda.max_memory_allocated(self.device)/1e9,3)
            self._last_epoch_diag['cuda_peak_reserved_gb']=round(torch.cuda.max_memory_reserved(self.device)/1e9,3)
        if not total:return {}
        return {key:float((value/max(total,1)).cpu()) for key,value in acc.items()}
    def _snapshot(self,path,phase,epoch,opt=None,val=None):
        save_ckpt(path,self.model,opt,self.cfg,self.meta,epoch,phase,val,self.memory_path)
    def train(self,phases=None,smoke=False,resume=None):
        phases=list(phases) if phases is not None else list(PHASE_ORDER)
        stop_at=self.cfg.get('training',{}).get('stop_after_phase')
        if stop_at is not None and stop_at not in PHASE_ORDER:
            raise ValueError(f'Invalid stop_after_phase: {stop_at}')
        if stop_at is not None and phases==list(PHASE_ORDER):
            phases=phases[:phases.index(stop_at)+1]
            print(f'FARE_PHASE_CAP stop_after={stop_at} phases={phases}',flush=True)
        resume_ck=None
        if resume:
            resume_ck=load_ckpt(resume,self.model,self.device,True)
            if resume_ck.get('metadata',{}).get('dataset')!=self.meta.get('dataset') or resume_ck.get('metadata',{}).get('target')!=self.meta.get('target'):raise RuntimeError('Resume dataset/target mismatch')
            if resume_ck['phase'] not in phases: raise ValueError(f'Resume phase {resume_ck["phase"]} not in requested phases {phases}')
            phases=phases[phases.index(resume_ck['phase']):]
            prev=self.run_dir/'phase_summary.json'
            if prev.exists():
                self.phase_history=json.loads(prev.read_text(encoding='utf-8'))
                completed=[item['phase'] for item in self.phase_history]
                if resume_ck['phase'] in completed:
                    phases=phases[1:]
                    prev_phase=resume_ck['phase']
                    phase_best=self.run_dir/f'{prev_phase}_best.pt'
                    if phase_best.exists():load_ckpt(phase_best,self.model,self.device)
                    resume_ck=None
        index_is_fresh=False
        for phase in phases:
            if phase not in PHASE_ORDER:raise ValueError(phase)
            if (phase in ('partial','full') and resume_ck is None and
                    self.cfg['training'].get('rollback_to_best_before_finetune',False)):
                previous=self.run_dir/'candidate_best.pt'
                if previous.exists():
                    load_ckpt(previous,self.model,self.device)
                    print(f'FARE_FINETUNE_ROLLBACK phase={phase} checkpoint={previous}',flush=True)
                    index_is_fresh=False
            self.model.set_phase(phase);opt=self._optimizer(phase)
            epochs=self.cfg['training']['phase_epochs'][phase]
            best=float('inf');stale=0;best_ep=0;best_ck=None;start_ep=1
            if resume_ck is not None and phase==resume_ck['phase']:
                if resume_ck.get('optimizer'):opt.load_state_dict(resume_ck['optimizer'])
                start_ep=int(resume_ck['epoch'])+1
                existing=self.run_dir/f'{phase}_best.pt'
                if existing.exists():
                    record=load_ckpt(existing)
                    best=float(record.get('validation',{}).get('total',float('inf')) if phase in ('retrieval','recovery') else record.get('validation',{}).get('forecast',float('inf')))
                    best_ep=int(record['epoch']);best_ck=existing
                print(f'FARE_RESUME phase={phase} start_epoch={start_ep}',flush=True)
            for ep in range(start_ep,epochs+1):
                t=time.time();refresh_seconds=0.0
                started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
                if self.cfg['experiment'].get('training_mode','full')!='no_memory' and not index_is_fresh:
                    t_refresh=time.time()
                    self.model.refresh(self.memory)
                    if self.device.type=='cuda':torch.cuda.synchronize(self.device)
                    refresh_seconds+=time.time()-t_refresh
                    index_is_fresh=True
                t_train=time.time()
                train=self._epoch(phase,opt,max_batches=2 if smoke else None)
                train_diag=self._last_epoch_diag.copy()
                if self.device.type=='cuda':torch.cuda.synchronize(self.device)
                train_seconds=time.time()-t_train
                if smoke and self.cfg['experiment'].get('training_mode','full')!='no_memory' and train.get('retrieval_valid_groups',0.) <= 0:
                    raise RuntimeError(
                        'FARE_SMOKE_NO_TRAIN_RETRIEVAL: zero legal TRAIN retrieval '
                        'groups; inspect TRAIN memory sampling and overlap exclusion'
                    )
                if self.cfg['experiment'].get('training_mode','full')!='no_memory':
                    t_refresh=time.time()
                    self.model.refresh(self.memory)
                    if self.device.type=='cuda':torch.cuda.synchronize(self.device)
                    refresh_seconds+=time.time()-t_refresh
                    index_is_fresh=True
                t_val=time.time()
                val=self._epoch(phase,None,max_batches=2 if smoke else None)
                val_diag=self._last_epoch_diag.copy()
                if self.device.type=='cuda':torch.cuda.synchronize(self.device)
                val_seconds=time.time()-t_val
                # Stage validation using canonical forecast metrics, rather than auxiliary losses.
                score=val['forecast'] if phase not in ('retrieval','recovery') or self.cfg['experiment'].get('training_mode')=='no_memory' else val['total']
                resource_rec={'utc_start':started_at,
                              'utc_end':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              'train_examples':int(train_diag.get('examples_processed',0)),
                              'val_examples':int(val_diag.get('examples_processed',0)),
                              'train_samples_per_second':round(train_diag.get('examples_processed',0)/max(train_seconds,1e-9),2),
                              'val_samples_per_second':round(val_diag.get('examples_processed',0)/max(val_seconds,1e-9),2),
                              'train_peak_cuda_allocated_gb':train_diag.get('cuda_peak_allocated_gb'),
                              'train_peak_cuda_reserved_gb':train_diag.get('cuda_peak_reserved_gb'),
                              'val_peak_cuda_allocated_gb':val_diag.get('cuda_peak_allocated_gb'),
                              'val_peak_cuda_reserved_gb':val_diag.get('cuda_peak_reserved_gb'),
                              'process_maxrss_gb':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,3),
                              'batch_size':int(self.cfg['training']['batch_size']),
                              'workers':int(self.cfg['training'].get('num_workers',0))}
                rec={'phase':phase,'epoch':ep,'train':train,'val':val,'seconds':round(time.time()-t,2),'score':score,
                     'train_seconds':round(train_seconds,2),'validation_seconds':round(val_seconds,2),
                     'memory_refresh_seconds':round(refresh_seconds,2),'resource':resource_rec}
                # Keep the original training-loss diagnostics beside the new
                # prediction/correction diagnostics in one analysis-friendly CSV.
                for split_stats,objective in ((train_diag,train),(val_diag,val)):
                    for metric in ('total','forecast','direct','state','innovation',
                                   'ranking','ranking_selected_error','ranking_oracle_error',
                                   'ranking_uniform_error','forecast_utility_rank',
                                   'utility_eligible_fraction','utility_positive_fraction',
                                   'coarse_utility_loss','coarse_utility_eligible_fraction',
                                   'coarse_utility_true_positive_fraction',
                                   'coarse_utility_pred_positive_fraction',
                                   'coarse_utility_sign_accuracy',
                                   'coarse_utility_oracle_positive_groups',
                                   'coarse_utility_expected_gain',
                                   'coarse_utility_oracle_gain',
                                   'coarse_utility_pairwise','coarse_utility_regret'):
                        if metric in objective:
                            split_stats[f'objective_{metric}']=objective[metric]
                diagnostic_record=append_diagnostics(self.run_dir,phase,ep,train_diag,val_diag,resource_rec)
                print('FARE_DIAG '+json.dumps(diagnostic_record,allow_nan=False),flush=True)
                print('FARE_RESOURCE_EPOCH '+json.dumps({'phase':phase,'epoch':ep,**resource_rec},allow_nan=False),flush=True)
                self.history.append(rec);print('FARE_EPOCH '+json.dumps(rec),flush=True)
                with (self.run_dir/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
                self._snapshot(self.run_dir/'last.pt',phase,ep,opt,val)
                if np.isfinite(score) and score<best-1e-6:
                    best=score;best_ep=ep;stale=0
                    best_ck=self.run_dir/f'{phase}_best.pt';self._snapshot(best_ck,phase,ep,opt,val)
                else:
                    stale+=1
                if stale>=self.cfg['training']['patience'] and ep>=self.cfg['training']['min_phase_epochs']:break
            if best_ck is None:raise RuntimeError(f'No valid checkpoint in phase {phase}')
            load_ckpt(best_ck,self.model,self.device)
            index_is_fresh=False
            if self.cfg['experiment'].get('training_mode','full')!='no_memory':
                self.model.refresh(self.memory)
                index_is_fresh=True
            stage_metrics=evaluate(self.model,self.memory,self.val_ds,self.device,self.cfg['training']['batch_size'],self.mean,self.std,self.run_dir/'stages'/phase,tag='val',save_arrays=self.cfg['evaluation']['save_stage_predictions'],mode=self.cfg['experiment'].get('training_mode','full'),refresh_memory=False)
            self.phase_history.append({
                'phase':phase,'epoch':best_ep,'selection_score':best,
                'val_MAE':stage_metrics['final']['MAE'],
                'val_RMSE':stage_metrics['final']['RMSE'],
                'val_directly_masked_MAE':stage_metrics['final']['directly_masked_MAE'],
                'val_base_MAE':stage_metrics['base']['MAE'],
                'val_candidate_MAE':stage_metrics['candidate']['MAE'],
                'val_gate_mean':stage_metrics['gate_mean'],
                'val_retrieval_valid_groups':stage_metrics.get('retrieval_diagnostics',{}).get('valid_group_fraction'),
                'val_candidate_MAE_gain':stage_metrics.get('corrections',{}).get('candidate_MAE_gain'),
                'val_final_MAE_gain':stage_metrics.get('corrections',{}).get('final_MAE_gain'),
                'val_final_harmful_fraction':stage_metrics.get('corrections',{}).get('final_harmful_fraction'),
                'val_gate_p90':stage_metrics.get('gate_statistics',{}).get('p90'),
                'val_directly_masked_RMSE':stage_metrics['final'].get('directly_masked_RMSE'),
            })
            json_write(self.run_dir/'phase_summary.json',self.phase_history)
            print('FARE_STAGE_DIAG '+json.dumps(self.phase_history[-1],allow_nan=False),flush=True)
            if phase in ('injection','partial','full') and stage_metrics['final']['MAE']<self.global_best:
                self.global_best=stage_metrics['final']['MAE'];self._snapshot(self.run_dir/'candidate_best.pt',phase,best_ep,None,stage_metrics['final'])
        if not (self.run_dir/'candidate_best.pt').exists():
            self._snapshot(self.run_dir/'candidate_best.pt',phases[-1],0,None,{'note':'intermediate stages only'})
        json_write(self.run_dir/'summary.json',{'phases':self.phase_history,'best_val_MAE':self.global_best,'candidate_checkpoint':str(self.run_dir/'candidate_best.pt'),'diagnostics_path':str(self.run_dir/'diagnostics'),'resource_monitor_prefix':self.resource_log_prefix})
        return self.run_dir/'candidate_best.pt'
