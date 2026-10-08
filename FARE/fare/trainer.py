from __future__ import annotations
import copy, csv, json, random, time
from pathlib import Path
import numpy as np,torch
from torch import nn
from .config import json_write,sha256
from .data import loader,to_device
from .losses import losses
from .evaluation import evaluate

PHASE_ORDER=['retrieval','recovery','injection','partial','full']

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def save_ckpt(path,model,opt,cfg,meta,epoch,phase,validation,memory_path=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={'version':'1.3.1','model':'FARE','phase':phase,'epoch':epoch,'state_dict':model.state_dict(),'optimizer':opt.state_dict() if opt is not None else None,'config':cfg,'metadata':meta,'validation':validation,'memory_path':str(memory_path) if memory_path else None,'rng':{'torch':torch.get_rng_state(),'numpy':np.random.get_state()}}
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
        json_write(self.run_dir/'run_manifest.json',{'cfg':cfg,'meta':meta,'memory_path':str(memory_path),'start_time':time.strftime('%Y-%m-%d %H:%M:%S')})
    def _optimizer(self,phase):
        base=[];new=[]
        for name,p in self.model.named_parameters():
            if not p.requires_grad:continue
            (base if name.startswith('stla.base.') else new).append(p)
        if not new and not base:raise RuntimeError('No trainable parameters')
        lr=self.cfg['training'];groups=[]
        if new:groups.append({'params':new,'lr':lr['new_lr']})
        if base:groups.append({'params':base,'lr':lr['partial_backbone_lr'] if phase=='partial' else lr['full_backbone_lr']})
        return torch.optim.AdamW(groups,weight_decay=lr['weight_decay'])
    def _epoch(self,phase,optimizer=None,max_batches=None):
        training=optimizer is not None; self.model.train(training)
        if not training:self.model.eval()
        ds=self.train_ds if training else self.val_ds
        it=loader(ds,self.cfg['training']['batch_size'],shuffle=training,workers=self.cfg['training']['num_workers'])
        acc={};total=0
        # All three quantities are derived from the actual forward path.
        # In v1.0.3 smoke train, retrieval_valid_groups was exactly zero.
        for j,batch in enumerate(it):
            if max_batches is not None and j>=max_batches:break
            bt=to_device(batch,self.device)
            if training:optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                out=self.model(bt,self.memory,mode=self.cfg['experiment'].get('training_mode','full'))
                for key in ('prediction','base_prediction','candidate_prediction',
                            'recovered_state','state_uncertainty','latent_innovation',
                            'innovation_uncertainty','retrieved_weights','final_gate'):
                    if key in out and not torch.isfinite(out[key]).all():
                        raise FloatingPointError(
                            f'FARE_NONFINITE_FORWARD phase={phase} '
                            f'batch={j} key={key}; refusing invalid validation/optimizer step'
                        )
                ls=losses(self.model,bt,out,self.memory,phase,self.mean,self.std,self.cfg)
                for key,value in ls.items():
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(
                            f'FARE_NONFINITE_LOSS phase={phase} batch={j} key={key}'
                        )
                if training:
                    ls['total'].backward()
                    # An apparently finite forward can still yield NaN
                    # gradients (e.g. sqrt(0)); catch this BEFORE AdamW.
                    params=[p for p in self.model.parameters() if p.requires_grad]
                    for name,p in self.model.named_parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            raise FloatingPointError(
                                f'FARE_NONFINITE_GRAD phase={phase} '
                                f'batch={j} parameter={name}'
                            )
                    nn.utils.clip_grad_norm_(params,self.cfg['training']['grad_clip'],
                                             error_if_nonfinite=True)
                    optimizer.step()
            bs=len(bt['x_obs']);total+=bs
            if 'retrieval_valid' in out:
                acc['retrieval_valid_groups']=acc.get('retrieval_valid_groups',0.)+float(out['retrieval_valid'].float().mean().detach())*bs
                acc['candidate_delta']=acc.get('candidate_delta',0.)+float((out['candidate_prediction']-out['base_prediction']).abs().mean().detach())*bs
                acc['final_gate_mean']=acc.get('final_gate_mean',0.)+float(out['final_gate'].mean().detach())*bs
            for k,v in ls.items():acc[k]=acc.get(k,0.)+float(v.detach())*bs
        return {k:v/max(total,1) for k,v in acc.items()}
    def _snapshot(self,path,phase,epoch,opt=None,val=None):
        save_ckpt(path,self.model,opt,self.cfg,self.meta,epoch,phase,val,self.memory_path)
    def train(self,phases=None,smoke=False,resume=None):
        phases=phases or PHASE_ORDER
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
        for phase in phases:
            if phase not in PHASE_ORDER:raise ValueError(phase)
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
                t=time.time()
                if self.cfg['experiment'].get('training_mode','full')!='no_memory':
                    self.model.refresh(self.memory)
                train=self._epoch(phase,opt,max_batches=2 if smoke else None)
                if smoke and self.cfg['experiment'].get('training_mode','full')!='no_memory' and train.get('retrieval_valid_groups',0.) <= 0:
                    raise RuntimeError(
                        'FARE_SMOKE_NO_TRAIN_RETRIEVAL: zero legal TRAIN retrieval '
                        'groups; inspect TRAIN memory sampling and overlap exclusion'
                    )
                if self.cfg['experiment'].get('training_mode','full')!='no_memory':
                    self.model.refresh(self.memory)
                val=self._epoch(phase,None,max_batches=2 if smoke else None)
                # Stage validation using canonical forecast metrics, rather than auxiliary losses.
                score=val['forecast'] if phase not in ('retrieval','recovery') or self.cfg['experiment'].get('training_mode')=='no_memory' else val['total']
                rec={'phase':phase,'epoch':ep,'train':train,'val':val,'seconds':round(time.time()-t,2),'score':score}
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
            if self.cfg['experiment'].get('training_mode','full')!='no_memory':
                self.model.refresh(self.memory)
            stage_metrics=evaluate(self.model,self.memory,self.val_ds,self.device,self.cfg['training']['batch_size'],self.mean,self.std,self.run_dir/'stages'/phase,tag='val',save_arrays=self.cfg['evaluation']['save_stage_predictions'],mode=self.cfg['experiment'].get('training_mode','full'))
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
            if phase in ('injection','partial','full') and stage_metrics['final']['MAE']<self.global_best:
                self.global_best=stage_metrics['final']['MAE'];self._snapshot(self.run_dir/'candidate_best.pt',phase,best_ep,None,stage_metrics['final'])
        if not (self.run_dir/'candidate_best.pt').exists():
            self._snapshot(self.run_dir/'candidate_best.pt',phases[-1],0,None,{'note':'intermediate stages only'})
        json_write(self.run_dir/'summary.json',{'phases':self.phase_history,'best_val_MAE':self.global_best,'candidate_checkpoint':str(self.run_dir/'candidate_best.pt')})
        return self.run_dir/'candidate_best.pt'
