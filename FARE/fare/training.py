"""Single Full-FARE staged training, Train-only counterfactual teacher refresh and resume."""
from pathlib import Path
import random,time,json,os
import numpy as np,torch
import torch.nn.functional as F
from .protocol import make_loader,device_batch
from .teacher import generate_teacher,save_teacher
from .config import dump_json,hash_file

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def forecasting_loss(out,batch,cfg):
    pred=out['prediction'];target=batch['target'];v=batch['future_valid'].bool()
    d=(pred-target).abs();valid=v.float();den=valid.sum().clamp_min(1)
    mae=(d*valid).sum()/den
    direct=batch['direct_mask'].bool()[:,None,:]&v
    masked=(d*direct).sum()/direct.sum().clamp_min(1)
    tail=F.smooth_l1_loss(pred,target,reduction='none')
    tail=(tail*valid).sum()/den
    return mae+cfg['training']['masked_weight']*masked+cfg['training']['tail_weight']*tail,{
        'mae':float(mae.detach()),'masked':float(masked.detach()),'tail':float(tail.detach())}

def auxiliary_loss(out,batch,teacher,device):
    if not teacher:return out['prediction'].sum()*0,{'teacher_examples':0}
    predicted=out['selection']['all_utility'];benefit_logits=out['selection']['all_benefit_logit'];pos=out['selection']['selected_positions']
    gain=out['gain_estimate'];r_losses=[];u_losses=[];replacement_losses=[];action_losses=[];examples=0
    for i,row in enumerate(batch['row_id'].tolist()):
        record=teacher.get(int(row))
        if record is None:continue
        v=record['valid'].to(device).bool()
        # A marginal-gain label belongs to the exact frozen Teacher evidence set.
        current=out['selection']['selected_ids'][i]
        if current.shape!=record['selected_ids'].shape:continue
        matched=(current==record['selected_ids'].to(device)).all(-1)
        src=out['shortlist_ids']
        if src is None:continue
        rp=int(record['replacement_pos'])
        if rp>=src.shape[-1]:continue
        matched=matched & src[i,:,rp].eq(record['replacement_candidate'].to(device))
        v=v & matched[:,None]
        if not v.any():continue
        slot=min(int(record['slot']),pos.shape[-1]-1)
        chosen=pos[i,:,slot]
        # benefit of the chosen episode averaged across its two evidence values
        util=predicted[i,torch.arange(len(chosen),device=device),chosen].mean(1)
        y=record['keep_gain'].to(device).float()
        y_r=record['risk_gain'].to(device).float()
        pos_idx=min(int(record['replacement_pos']),predicted.shape[2]-1)
        replacement_target=record['replace_gain'].to(device).float()
        replacement_utility=predicted[i,:,pos_idx].mean(1)
        replacement_difference=replacement_utility-util
        benefit=benefit_logits[i,torch.arange(len(chosen),device=device),chosen].mean(1)
        replacement_benefit=benefit_logits[i,:,pos_idx].mean(1)-benefit
        replacement_losses.append(F.smooth_l1_loss(replacement_difference[v],replacement_target[v])+
                                   0.2*F.binary_cross_entropy_with_logits(replacement_benefit[v],(replacement_target[v]>0).float()))
        action_target=record['action_alpha'].to(device).float()
        action_losses.append(F.smooth_l1_loss(out['alpha'][i][v],action_target[v]))
        u_losses.append(F.smooth_l1_loss(util[v],y[v])+0.2*F.binary_cross_entropy_with_logits(benefit[v],(y[v]>0).float()))
        r_losses.append(F.smooth_l1_loss(gain[i][v],y_r[v])+0.2*F.binary_cross_entropy_with_logits(gain[i][v],(y_r[v]>0).float()))
        examples+=1
    if not examples:return gain.sum()*0,{'teacher_examples':0}
    a=torch.stack(u_losses).mean()+torch.stack(replacement_losses).mean()
    b=torch.stack(r_losses).mean()+0.5*torch.stack(action_losses).mean()
    return a+b,{'teacher_examples':examples,'utility_loss':float(a.detach()),'risk_loss':float(b.detach()),
                'replacement_loss':float(torch.stack(replacement_losses).mean().detach()),
                'action_loss':float(torch.stack(action_losses).mean().detach())}

def checkpoint(path,model,optimizer,cfg,run_state,metadata):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'version':'3.0.2','state_dict':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                'config':cfg,'training_state':run_state,'metadata':metadata,
                'torch_rng':torch.get_rng_state(),
                'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},p)

def fit(model,memory,train_ds,val_ds,device,cfg,run_dir,metadata,resume=None):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=float(cfg['training']['lr']),
                          weight_decay=float(cfg['training']['weight_decay']))
    stage_epochs=cfg['training']['stage_epochs']
    stages=list(stage_epochs)
    teacher=None;start_stage=0;start_ep=0;best=float('inf');steps=0
    if resume:
        ck=torch.load(resume,map_location=device,weights_only=False)
        model.load_state_dict(ck['state_dict'],strict=True);opt.load_state_dict(ck['optimizer_state'])
        if 'torch_rng' in ck:torch.set_rng_state(ck['torch_rng'].cpu())
        if 'numpy_rng' in ck:np.random.set_state(ck['numpy_rng'])
        if 'python_rng' in ck:random.setstate(ck['python_rng'])
        if ck.get('cuda_rng') is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(ck['cuda_rng'])
        state=ck['training_state'];start_stage=state['stage_idx'];start_ep=state['stage_epoch']+1
        if start_ep>=int(stage_epochs[stages[start_stage]]):start_stage+=1;start_ep=0
        best=state['best_val'];steps=state.get('steps',0)
    log_path=run_dir/'train_metrics.jsonl'
    for stage_idx in range(start_stage,len(stages)):
        stage=stages[stage_idx]
        epoch_first=start_ep if stage_idx==start_stage else 0
        if stage in ('utility','risk','joint'):
            model.eval()
            teacher=generate_teacher(model,memory,train_ds,device,cfg['teacher']['max_queries'],
                                     cfg['teacher']['batch_size'],cfg['memory']['purge_steps'])
            tfile=run_dir/f'teacher_{stage}.pt'
            save_teacher(teacher,tfile,{'stage':stage,'seed':cfg['experiment']['seed'],
                                         'source_checkpoint':'in-run snapshot','count':len(teacher)})
            dump_json(run_dir/f'teacher_{stage}.json',{'stage':stage,'rows':len(teacher),'sha256':hash_file(tfile)})
        for epoch in range(epoch_first,int(stage_epochs[stage])):
            refresh_period=int(cfg['teacher'].get('refresh_every_epochs',0))
            if stage in ('utility','risk','joint') and refresh_period>0 and epoch>epoch_first and epoch%refresh_period==0:
                model.eval()
                teacher=generate_teacher(model,memory,train_ds,device,cfg['teacher']['max_queries'],
                                         cfg['teacher']['batch_size'],cfg['memory']['purge_steps'])
                tfile=run_dir/f'teacher_{stage}_epoch{epoch+1}.pt'
                save_teacher(teacher,tfile,{'stage':stage,'epoch':epoch+1,'count':len(teacher)})
                dump_json(tfile.with_suffix('.json'),{'stage':stage,'rows':len(teacher),'sha256':hash_file(tfile)})
            model.train();loss_total=0.0;count=0
            loader=make_loader(train_ds,cfg['training']['batch_size'],shuffle=True,workers=cfg['training']['workers'])
            for raw in loader:
                bt=device_batch(raw,device)
                opt.zero_grad(set_to_none=True)
                augmentation='full'
                if stage in ('evidence','integration'):
                    r=random.random()
                    if r<0.04:augmentation='no_residual'
                    elif r<0.08:augmentation='no_evolution'
                    elif r<0.10:augmentation='no_support'
                out=model(bt,memory,intervention=augmentation,purge=cfg['memory']['purge_steps'],return_details=False)
                loss,diag=forecasting_loss(out,bt,cfg)
                if stage in ('utility','risk','joint') and teacher:
                    aux,aux_d=auxiliary_loss(out,bt,teacher,device)
                    loss=loss+cfg['training']['utility_weight']*aux
                else:aux_d={'teacher_examples':0}
                if not torch.isfinite(loss):raise FloatingPointError('FARE training loss is nonfinite')
                loss.backward()
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),cfg['training']['grad_clip'])
                opt.step();steps+=1;loss_total+=float(loss.detach());count+=1
                if steps%int(cfg['training']['log_every'])==0:
                    print(f'FARE_TRAIN stage={stage} epoch={epoch+1} step={steps} loss={float(loss.detach()):.5f} '
                          f'mae={diag["mae"]:.5f} alpha={float(out["alpha"].mean().detach()):.3f} '
                          f'null={float(out["selection"]["null"].mean().detach()):.3f}',flush=True)
            val=validate_loss(model,memory,val_ds,device,cfg)
            state={'stage_idx':stage_idx,'stage':stage,'stage_epoch':epoch,'best_val':best,'steps':steps}
            rec={'time':time.strftime('%Y-%m-%dT%H:%M:%S'),'stage':stage,'epoch':epoch+1,
                 'train_loss':loss_total/max(count,1),'val_mae_norm':val,'steps':steps,**aux_d}
            with open(log_path,'a') as f:f.write(json.dumps(rec)+'\n')
            if val<best:
                best=val;state['best_val']=best
                checkpoint(run_dir/'best.pt',model,opt,cfg,state,metadata)
            checkpoint(run_dir/'last.pt',model,opt,cfg,state,metadata)
            print('FARE_EPOCH '+json.dumps(rec),flush=True)
    dump_json(run_dir/'training_complete.json',{'best_val_normalized':best,'steps':steps,'time':time.strftime('%Y-%m-%dT%H:%M:%S')})
    print('FARE_TRAIN_COMPLETE',flush=True)
    return run_dir/'best.pt'

@torch.no_grad()
def validate_loss(model,memory,ds,device,cfg):
    model.eval();num=0.0;den=0
    for raw in make_loader(ds,cfg['training']['batch_size'],workers=cfg['training']['workers']):
        bt=device_batch(raw,device)
        out=model(bt,memory,purge=cfg['memory']['purge_steps'],return_details=False)
        valid=bt['future_valid'].bool()
        num+=float(((out['prediction']-bt['target']).abs()*valid).sum())
        den+=int(valid.sum())
    return num/max(den,1)
