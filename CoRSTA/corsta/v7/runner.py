"""Single-allocation stage engine for v7. Every stage runs in an isolated process.

Train targets: only train_x_full_norm.npy; validation/test never touch it.
No TEST outcomes are used for selection; TEST starts after a frozen lock manifest.
"""
from __future__ import annotations
import json,os,random,hashlib,time,math,resource,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from .model import SupportAwareTCRR,make_neighbours
from .graph import load_or_build_graph
from .replay import ReplayDataset,prepare_replay
from .sampling import replay_train_loader
from .storage import cache_teacher,file_manifest,CachedDataset
from ..stla import load_pretrained_stla,MaskAwareSTLAformer
from ..config import file_sha256,cache_path
from ..data import SupportSplit

STAGES=('s0','cache','replay','normal','c1','c2','joint_c12','lock','test','report')


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def atomic_save(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial');torch.save(obj,tmp);os.replace(tmp,path)


def atomic_json(data,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(data,indent=2,default=str));os.replace(tmp,path)


def as_device(batch,device):
    return {k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}


def loss_masked(pred,y,mask):
    if pred.shape!=y.shape or pred.shape!=mask.shape:raise ValueError('Prediction/target mask shapes differ')
    w=mask.float();return ((pred.float()-y.float()).abs()*w).sum()/w.sum().clamp_min(1)


def selection_indices(ds):
    """Chronological, paired-fault-group-safe Selection/Locked Check split.

    All masking variants of the same window remain in the same partition.
    A 24-step history+horizon gap prevents boundary overlap.
    """
    starts=np.asarray(ds.window_starts[:len(ds)],dtype=np.int64)
    unique=np.unique(starts)
    if len(unique)<3:raise RuntimeError('Validation needs at least three distinct windows')
    cut_sel=unique[min(len(unique)-1,max(0,int(len(unique)*.60)-1))]
    cut_lock=unique[min(len(unique)-1,max(1,int(len(unique)*.80)))]
    selection=np.flatnonzero(starts<=cut_sel).tolist()
    locked=np.flatnonzero(starts>=max(cut_lock,int(cut_sel)+24)).tolist()
    if not selection or not locked:raise RuntimeError('Chronological validation split empty')
    if int(starts[selection].max())+24 > int(starts[locked].min()):
        raise RuntimeError('Selection and Locked Check windows overlap')
    return selection,locked


def loader(ds,batch_size,workers,shuffle=False,subset=None):
    x=Subset(ds,subset) if subset is not None else ds
    return DataLoader(x,batch_size=batch_size,shuffle=shuffle,num_workers=workers,
        pin_memory=torch.cuda.is_available(),persistent_workers=(workers>0),prefetch_factor=2 if workers>0 else None)


def make_teacher(args,device,s0=None):
    base,bc=load_pretrained_stla(Path(args.root),Path(args.backbone_checkpoint),device)
    teacher=MaskAwareSTLAformer(base).to(device)
    if int(base.num_nodes)!=int(args.nodes):raise RuntimeError('Backbone nodes differ from dataset')
    if s0 is not None:teacher.mask_proj.load_state_dict(s0['mask_proj'])
    teacher.eval();return teacher,bc


def s0_step(args,ds_train,ds_val,device):
    out=args.run/'s0.pt'
    teacher,bc=make_teacher(args,device)
    for p in teacher.base.parameters():p.requires_grad_(False)
    for p in teacher.mask_proj.parameters():p.requires_grad_(True)
    first,_=selection_indices(ds_val)
    if device.type=='cuda' and os.environ.get('CORSTA_BOOTSTRAP_AUTO_BATCH','1')=='1':
        from ..batching import select_batch
        class _TeacherProbe(nn.Module):
            def __init__(self,net):super().__init__();self.net=net
            def forward(self,b):
                pred,_=self.net(b['x'],b['mask'],b['tod'],b['dow'])
                return {'refined':pred,'c1_posthoc':pred}
        args.batch_size,attempts=select_batch(_TeacherProbe(teacher),ds_train,device,args.amp,initial=64)
        atomic_json({'phase':'s0','batch':args.batch_size,'attempts':attempts},args.run/'batch_probe_s0.json')
    tl=loader(ds_train,args.batch_size,args.workers,True)
    vl=loader(ds_val,args.eval_batch_size,0,subset=first)
    optim=torch.optim.AdamW(teacher.mask_proj.parameters(),lr=args.s0_lr)
    mean=float(bc['mean']);std=float(bc['std'])
    # S0 checkpoints are independent of all older TCRR versions. Interrupted S0
    # restarts at the last completed epoch rather than repeating ~1 hour of work.
    last=args.run/'last_s0.pt'
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda' and args.amp))
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved.get('version')!='7.0.1' or saved.get('backbone_sha256')!=file_sha256(args.backbone_checkpoint):
            raise RuntimeError('S0 resume identity does not match official Backbone')
        teacher.mask_proj.load_state_dict(saved['current']);optim.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        best=float(saved['best']);best_epoch=int(saved['best_epoch']);best_state=saved['best_state']
        initial_epoch=int(saved['epoch'])+1
        torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        if device.type=='cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print(f'V7_S0_RESUME from_epoch={initial_epoch} best={best:.6f}',flush=True)
    else:
        teacher.eval();best_sum=0.;best_count=0
        with torch.no_grad():
            for raw in vl:
                b=as_device(raw,device)
                pred,_=teacher(b['x'],b['mask'],b['tod'],b['dow'])
                valid=b['future_valid'].float()
                best_sum+=float((((pred.float()*std+mean)-b['y_raw']).abs()*valid).sum())
                best_count+=int(valid.sum())
        best=best_sum/max(1,best_count)
        best_state=copy.deepcopy(teacher.mask_proj.state_dict());best_epoch=0;initial_epoch=1
        print(f'V7_S0_BASELINE_SELECTION_MAE={best:.6f}',flush=True)
    for ep in range(initial_epoch,args.epochs['s0']+1):
        for is_train,dl in ((True,tl),(False,vl)):
            teacher.train(is_train);teacher.base.eval();s=0.;c=0
            for raw in dl:
                b=as_device(raw,device)
                with torch.set_grad_enabled(is_train):
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=scaler.is_enabled()):
                        pred,_=teacher(b['x'],b['mask'],b['tod'],b['dow'])
                        loss=loss_masked(pred,b['y_norm'],b['future_valid'])
                    if is_train:
                        optim.zero_grad(set_to_none=True)
                        if scaler.is_enabled():
                            scaler.scale(loss).backward();scaler.unscale_(optim)
                            nn.utils.clip_grad_norm_(teacher.mask_proj.parameters(),5)
                            scaler.step(optim);scaler.update()
                        else:
                            loss.backward();optim.step()
                valid=b['future_valid'].float();s+=float(((pred.float()*std+mean-b['y_raw']).abs()*valid).sum().detach());c+=int(valid.sum())
            score=s/max(c,1)
            if not is_train and score<best:
                best=score;best_epoch=ep;best_state=copy.deepcopy(teacher.mask_proj.state_dict())
            print(f'V7_S0_EPOCH epoch={ep} phase={"train" if is_train else "selection"} MAE={score:.6f} best={best:.6f}',flush=True)
        atomic_save({'version':'7.0.1','epoch':ep,'current':teacher.mask_proj.state_dict(),
                     'optimizer':optim.state_dict(),'scaler':scaler.state_dict(),
                     'best':best,'best_state':best_state,'best_epoch':best_epoch,
                     'backbone_sha256':file_sha256(args.backbone_checkpoint),
                     'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
                     'python_rng':random.getstate(),
                     'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None},last)
    atomic_save({'version':'7.0.1','mask_proj':best_state,'backbone_sha256':file_sha256(args.backbone_checkpoint),
                 'best_selection_mae':best,'epoch':best_epoch,'target':args.target,'dataset':args.dataset},out)
    print('V7_S0_PASS',out,flush=True)


def summarize_train_response(records):
    """TRAIN-cache diagnostic names match the actual serialized arrays."""
    keys=('normal','c2','cf')
    missing=[k for k in keys if k not in records]
    if missing:raise RuntimeError(f'Teacher TRAIN cache missing keys: {missing}')
    return {k:float(np.abs(np.asarray(records[k])).mean()) for k in keys}


def train_cache(args,device,stage='train',severity=None):
    """Fresh v7: only own teacher caches under this run; never read old model outputs."""
    source=cache_path(args.config,'evaluation',severity) if severity else cache_path(args.config,'formal')
    ds=SupportSplit(source,'test' if severity else stage,
                    limit=args.limit('test' if severity else stage),cf_training=not severity and stage=='train')
    adapter=torch.load(args.run/'s0.pt',map_location='cpu',weights_only=False)
    if adapter['backbone_sha256']!=file_sha256(args.backbone_checkpoint):raise RuntimeError('S0 backbone identity mismatch')
    # Validate complete v7-owned cache BEFORE loading the expensive backbone.
    from .storage import open_existing_teacher
    root=args.run/'teacher_cache'/('test_s'+str(severity) if severity else stage)
    identity=file_manifest(ds,'test' if severity else stage,file_sha256(args.run/'s0.pt'),args.groups)
    records=open_existing_teacher('test' if severity else stage,root,identity,len(ds),args.nodes,args.horizon,args.model_dim)
    if records is not None:
        if stage=='train' and severity is None:
            stat_path=args.run/'counterfactual_target_strength.json'
            if not stat_path.exists():
                strength=summarize_train_response(records)
                strength['note']='Train-only teacher response magnitudes; not deployable targets'
                atomic_json(strength,stat_path)
        bc=torch.load(args.backbone_checkpoint,map_location='cpu',weights_only=False)
        print(f'V7_TEACHER_REUSE_NO_STLA_LOAD split={stage} severity={severity}',flush=True)
        return ds,records,bc
    teacher,bc=make_teacher(args,device,adapter)
    # Train-only outputs depend on the dataset, S0 identity, and grouping protocol.
    root=args.run/'teacher_cache'/('test_s'+str(severity) if severity else stage)
    try:
        records=cache_teacher('test' if severity else stage,ds,teacher,root,device,
            identity,batch_size=args.teacher_batch,workers=0)
    finally:
        del teacher
        if device.type=='cuda':torch.cuda.empty_cache()
    if stage=='train' and severity is None:
        stat_path=args.run/'counterfactual_target_strength.json'
        if not stat_path.exists():
            strength=summarize_train_response(records)
            strength['note']='Train-only teacher response magnitudes; not model gains or deployable oracles'
            atomic_json(strength,stat_path)
    return ds,records,bc


def graph_file(args):return args.run/'train_only_neighbours.pt'


def graph_and_memory(args,device):
    train,records,bc=train_cache(args,device,'train')
    graph=graph_file(args)
    source=train.root/'train_x_full_norm.npy'
    meta={'file_sha256':file_sha256(source),'queries':len(train),'neighbors':args.graph_neighbours,'nodes':args.nodes,
          'graph_algorithm_sha256':file_sha256(Path(__file__).with_name('model.py'))}
    if graph.exists():
        saved=torch.load(graph,map_location='cpu',weights_only=False)
        if not isinstance(saved,dict) or saved.get('identity')!=meta:
            raise RuntimeError('Train graph has stale identity; use a new run directory')
        neigh=saved['neighbours'];neigh_valid=saved['neighbours_valid']
    else:
        xx=np.load(source,mmap_mode='r')
        selected=np.linspace(0,len(train)-1,min(4096,len(train)),dtype=int)
        sample=np.nan_to_num(np.array(xx[selected],dtype=np.float32,copy=True))
        src=np.asarray(train.original_valid[selected],bool)|np.asarray(train.observed[selected],bool)
        neigh,neigh_valid=make_neighbours(torch.from_numpy(sample),args.graph_neighbours,
                                        torch.from_numpy(src.copy()))
        atomic_save({'identity':meta,'neighbours':neigh,'neighbours_valid':neigh_valid},graph)
    road=load_or_build_graph(args.root,args.target,args.dataset,train.root,args.run/'graph',args.road_k)
    return None,{'functional':neigh,'functional_valid':neigh_valid,'road':road},bc


def new_network(args,device,graph,bc):
    return SupportAwareTCRR(model_dim=args.model_dim,horizon=args.horizon,history=args.history,
       neighbours=graph['functional'],functional_valid=graph['functional_valid'],
       road_graph=graph['road'],graph_mode=args.graph_mode,
       hidden=args.hidden,bound=args.bound,top_k=args.k,
       support_control=args.support_control,horizon_graph=args.horizon_graph,
       reliability_graph=args.reliability_graph).to(device)


def load_network(args,device,graph,bc,previous=None):
    net=new_network(args,device,graph,bc)
    if previous:
        ck=torch.load(args.run/f'best_{previous}.pt',map_location='cpu',weights_only=False)
        if ck['version']!='7.0.1' or ck['backbone_sha256']!=file_sha256(args.backbone_checkpoint):
            raise RuntimeError('Previous stage checkpoint identity mismatch')
        net.load_state_dict(ck['model'],strict=True)
    return net


def phase_prediction(out,phase):
    return out['normal' if phase=='normal' else ('c1' if phase=='c1' else 'c12')]


def evaluate_modes(args,net,dl,memory,device,mean,std,metrics=True):
    """All five predictions from the identical same-time batch and same checkpoint."""
    net.eval();stats={m:[0.,0.,0.,0.,0.,0.,0.]  for m in ('backbone','normal','c1','c12')}
    with torch.inference_mode():
        for raw in dl:
            b=as_device(raw,device)
            o=net.forward_progressive(b['cached_base'],b['cached_latent'],b['x'],b['mask'],
                b['candidate_normal'],b['candidate_cf'],candidate_scores=b['candidate_scores'],
                candidate_coverage=b['candidate_coverage'],candidate_valid=b['candidate_valid'],
                candidate_support_raw=b['candidate_support_raw'])
            for mode in stats:
                p=o[mode].float()*std+mean
                mask=b['future_valid'].bool();err=p-b['y_raw']
                stats[mode][0]+=float((err.abs()*mask).sum());stats[mode][1]+=float((err.square()*mask).sum());stats[mode][2]+=int(mask.sum())
                # Direct mask is a known evaluation index, never a model input.
                direct=b['direct_mask'].bool()
                if direct.ndim==2:direct=direct.unsqueeze(1)
                if direct.shape not in ((len(mask),1,mask.shape[-1]),mask.shape):
                    raise ValueError(f'Invalid direct_mask shape: {direct.shape}')
                dm=direct & mask
                stats[mode][3]+=float((err.abs()*dm).sum());stats[mode][4]+=int(dm.sum())
                stats[mode][5]+=float((err[:,-1].abs()*mask[:,-1]).sum());stats[mode][6]+=int(mask[:,-1].sum())
    return {k:{'mae':v[0]/max(1,v[2]),'rmse':math.sqrt(v[1]/max(1,v[2])),
                'masked_mae':v[3]/max(1,v[4]),'h12_mae':v[5]/max(1,v[6])}
            for k,v in stats.items()}


def fit_stage(args,phase,device):
    prev={'normal':None,'c1':'normal','c2':'c1','joint_c12':'c2'}[phase]
    memory,graph,bc=graph_and_memory(args,device)
    src=cache_path(args.config,'formal')
    train_ds=SupportSplit(src,'train',limit=args.limit('train'))
    val_ds=SupportSplit(src,'val',limit=args.limit('val'))
    train_records={k:np.load(args.run/'teacher_cache/train'/f'train_{k}.npy',mmap_mode='r')
                   for k in ('base','latent','c2')}
    val_records={k:np.load(args.run/'teacher_cache/val'/f'val_{k}.npy',mmap_mode='r')
                 for k in ('base','latent')}
    tl=replay_train_loader(ReplayDataset(train_ds,train_records,args.run/'paired_replay/train',args.replay_chunk),args.batch_size,args.workers)
    selection,_=selection_indices(val_ds)
    vl=loader(ReplayDataset(val_ds,val_records,args.run/'paired_replay/val',args.replay_chunk),args.eval_batch_size,0,subset=selection)
    net=load_network(args,device,graph,bc,prev)
    net.freeze_stage(phase); trainable=[p for p in net.parameters() if p.requires_grad]
    if device.type=='cuda' and os.environ.get('CORSTA_BOOTSTRAP_AUTO_BATCH','1')=='1':
        from ..batching import select_batch
        class _LegacyProbe(nn.Module):
            def __init__(self,net,phase):super().__init__();self.net=net;self.phase=phase
            def forward(self,b):
                stage={'normal':0,'c1':1,'c2':2,'joint_c12':2}[self.phase]
                out=self.net.forward_progressive(b['cached_base'],b['cached_latent'],
                    b['x'],b['mask'],b['candidate_normal'],b['candidate_cf'],
                    candidate_scores=b['candidate_scores'],candidate_coverage=b['candidate_coverage'],
                    candidate_valid=b['candidate_valid'],candidate_support_raw=b['candidate_support_raw'],
                    max_stage=stage)
                pred=out['normal' if self.phase=='normal' else ('c1' if self.phase=='c1' else 'c12')]
                return {'refined':pred,'c1_posthoc':pred}
        args.batch_size,attempts=select_batch(_LegacyProbe(net,phase),tl.dataset,
                                              device,args.amp,initial=64)
        atomic_json({'phase':phase,'batch':args.batch_size,'attempts':attempts},
                    args.run/f'batch_probe_{phase}.json')
        tl=replay_train_loader(tl.dataset,args.batch_size,args.workers)
    if phase=='joint_c12':
        small=[];large=[]
        for name,p in net.named_parameters():
            if not p.requires_grad:continue
            (large if name.startswith(('response_encoder.','query_proj.','key_proj.',
                             'edge_proj.','horizon_proj.','source_reliability.',
                             'spatial_fusion.','c2_')) else small).append(p)
        optim=torch.optim.AdamW([{'params':small,'lr':args.learning_rate*.1},
                                 {'params':large,'lr':args.learning_rate}],weight_decay=1e-4)
    else:
        optim=torch.optim.AdamW(trainable,lr=args.learning_rate,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda' and args.amp))
    mean,std=float(bc['mean']),float(bc['std'])
    # Epoch 0 is the frozen previous-stage output because the new output head
    # is zero-initialized. A weak stage must not silently worsen Selection.
    prev_key={'normal':'backbone','c1':'normal','c2':'c1','joint_c12':'c12'}[phase]
    current_key={'normal':'normal','c1':'c1','c2':'c12','joint_c12':'c12'}[phase]
    last=args.run/f'last_{phase}.pt'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck.get('version')!='7.0.1' or ck.get('phase')!=phase or ck.get('backbone_sha256')!=file_sha256(args.backbone_checkpoint):
            raise RuntimeError('Last checkpoint does not match this stage or Backbone')
        if not (args.run/f'best_{phase}.pt').exists():raise RuntimeError('Missing best stage checkpoint on resume')
        net.load_state_dict(ck['model'],strict=True)
        optim.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        best=float(ck['best']);best_epoch=int(ck['best_epoch']);wait=int(ck['wait']);initial=float(ck['initial'])
        start_epoch=int(ck['epoch'])+1
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        print(f'V7_RESUME_PASS stage={phase} next_epoch={start_epoch} best_epoch={best_epoch}',flush=True)
    else:
        baseline_result=evaluate_modes(args,net,vl,memory,device,mean,std)
        initial=baseline_result[prev_key]['mae']
        best=baseline_result[current_key]['mae'];best_epoch=0;wait=0;start_epoch=1
        atomic_save({'version':'7.0.1','model':net.state_dict(),'phase':phase,'epoch':0,
                     'best_selection_mae':best,'backbone_sha256':file_sha256(args.backbone_checkpoint),
                     'dataset':args.dataset,'target':args.target,'seed':args.seed,
                     'config':args.public_config},args.run/f'best_{phase}.pt')
    for ep in range(start_epoch,args.epochs[phase]+1):
        tic=time.perf_counter();net.train();summ=0.;count=0;aux=0.;accum=args.grad_accum
        optim.zero_grad(set_to_none=True)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for ix,raw in enumerate(tl,1):
            b=as_device(raw,device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=scaler.is_enabled()):
                o=net.forward_progressive(b['cached_base'],b['cached_latent'],b['x'],b['mask'],
                      b['candidate_normal'],b['candidate_cf'],candidate_scores=b['candidate_scores'],
                      candidate_coverage=b['candidate_coverage'],candidate_valid=b['candidate_valid'],
                      candidate_support_raw=b['candidate_support_raw'],
                      shuffled_negatives=(phase=='c1' and ix%4==0),
                      max_stage={'normal':0,'c1':1,'c2':2,'joint_c12':2}[phase])
                pred=phase_prediction(o,phase)
                # v5.3-compatible raw-unit forecast-first loss.
                err=pred.float()*std+mean-b['y_raw'].float()
                forecast=loss_masked(pred.float()*std+mean,b['y_raw'],b['future_valid'])
                rmse=torch.sqrt(((err.square()*b['future_valid'].float()).sum()/b['future_valid'].sum().clamp_min(1))+1e-8)
                loss=forecast+0.05*rmse
                response=forecast.new_zeros(())
                if phase in ('c1','joint_c12'):
                    from .losses import cfrm_auxiliary
                    response=cfrm_auxiliary(o,b,mean,std,wrong_pair=('cf_wrong_increment' in o))
                    loss=loss+(0.2 if phase=='joint_c12' else 1.0)*response
                    if phase=='joint_c12':
                        response=response+loss_masked(o['delta2'],b['cached_c2'],b['future_valid'])
                        loss=loss+args.response_weight*loss_masked(o['delta2'],b['cached_c2'],b['future_valid'])
                elif phase=='c2':
                    response=loss_masked(o['delta2'],b['cached_c2'],b['future_valid'])
                    loss=loss+args.response_weight*response
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {phase} ep={ep} batch={ix}')
            step_loss=loss/accum
            if scaler.is_enabled():scaler.scale(step_loss).backward()
            else:step_loss.backward()
            if ix%accum==0 or ix==len(tl):
                if scaler.is_enabled():scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(trainable,5.)
                if scaler.is_enabled():scaler.step(optim);scaler.update()
                else:optim.step()
                optim.zero_grad(set_to_none=True)
            valid=int(b['future_valid'].sum());summ+=float(forecast.detach())*valid;aux+=float(response.detach())*valid;count+=valid
        result=evaluate_modes(args,net,vl,memory,device,mean,std)
        key={'normal':'normal','c1':'c1','c2':'c12','joint_c12':'c12'}[phase]
        metric=result[key]['mae'];prev_key={'normal':'backbone','c1':'normal','c2':'c1','joint_c12':'c12'}[phase]
        gain=result[prev_key]['mae']-metric
        if metric<best-1e-7:
            best=metric;best_epoch=ep;wait=0
            ck={'version':'7.0.1','model':net.state_dict(),'phase':phase,'epoch':ep,'best_selection_mae':best,
                'backbone_sha256':file_sha256(args.backbone_checkpoint),'dataset':args.dataset,'target':args.target,
                'seed':args.seed,'config':args.public_config}
            atomic_save(ck,args.run/f'best_{phase}.pt')
        else:wait+=1
        peak=torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else 0
        print(f'V7_EPOCH stage={phase} epoch={ep} train_mae_raw={summ/max(count,1):.6f} '
              f'response_loss={aux/max(count,1):.6f} val_mae={metric:.6f} '
              f'val_prev={result[prev_key]["mae"]:.6f} marginal_gain={gain:.6f} '
              f'best={best:.6f} throughput={len(train_ds)/max(time.perf_counter()-tic,1):.2f}q/s '
              f'gpu_peak_gib={peak:.2f}',flush=True)
        atomic_save({'version':'7.0.1','phase':phase,'epoch':ep,'model':net.state_dict(),
                     'optimizer':optim.state_dict(),'scaler':scaler.state_dict(),
                     'best':best,'best_epoch':best_epoch,'wait':wait,'initial':initial,
                     'backbone_sha256':file_sha256(args.backbone_checkpoint),
                     'torch_rng':torch.get_rng_state(),
                     'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                     'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},last)
        if wait>=args.patience:break
    if not (args.run/f'best_{phase}.pt').exists():raise RuntimeError(f'No {phase} checkpoint created')
    atomic_json({'phase':phase,'best_selection_mae':best,'previous_selection_mae':initial,
                 'best_epoch':best_epoch},args.run/f'selection_{phase}.json')
    print(f'V7_STAGE_PASS stage={phase} best_epoch={best_epoch} selection_mae={best:.6f} '
          f'previous_selection_mae={initial:.6f} gain={initial-best:.6f}',flush=True)


def evaluate_run(args,device,split='val',severity=None):
    memory,graph,bc=graph_and_memory(args,device)
    ds,records,_=train_cache(args,device,split,severity)
    net=load_network(args,device,graph,bc,'joint_c12');net.eval()
    replay_dir=prepare_replay(args,device,'test' if severity is not None else split,severity=severity)
    dataset=ReplayDataset(ds,records,replay_dir,args.replay_chunk)
    if severity:
        indices=None
    else:
        _,indices=selection_indices(ds)
    dl=loader(dataset,args.eval_batch_size,0,subset=indices)
    return evaluate_modes(args,net,dl,memory,device,float(bc['mean']),float(bc['std']))


def locked_check(args,device):
    result=evaluate_run(args,device,'val')
    atomic_json({'version':'7.0.1','dataset':args.dataset,'target':args.target,
                 'seed':args.seed,'result':result,
                 'note':'Locked subset is read once after Selection, never used to choose variants'},
                args.run/'locked_check.json')
    atomic_json({'version':'7.0.1','selected_mode':'c12',
                 'model':str(args.run/'best_joint_c12.pt'),
                 'stage_checkpoints':[str(args.run/f'best_{s}.pt')
                                      for s in ('normal','c1','c2','joint_c12')],
                 'backbone_sha256':file_sha256(args.backbone_checkpoint)},
                args.run/'model_lock.json')
    print('V7_LOCK_PASS '+json.dumps(result),flush=True)


def test_severity(args,device,severity):
    if not (args.run/'model_lock.json').exists():raise RuntimeError('TEST without model lock forbidden')
    result=evaluate_run(args,device,severity=severity)
    chosen=json.loads((args.run/'model_lock.json').read_text())['selected_mode']
    result['selected']=result[chosen].copy()
    atomic_json({'version':'7.0.1','target':args.target,'severity':severity,'metrics':result},args.run/'test'/f's{severity}.json')
    print(f'V7_TEST_PASS severity=s{severity} '+json.dumps(result),flush=True)


def report(args):
    test={}
    for severity in args.severities:
        p=args.run/'test'/f's{severity}.json'
        if not p.exists():raise FileNotFoundError(f'Cannot report partial TEST: {p}')
        test['s'+str(severity)]=json.loads(p.read_text())['metrics']
    modes=('backbone','normal','c1','c12','selected')
    result={'version':'7.0.1','target':args.target,'dataset':args.dataset,'seed':args.seed,'severities':test,
            'mean_four_severities':{m:{met:sum(test[s][m][met] for s in test)/len(test)
                                         for met in ('mae','rmse','masked_mae','h12_mae')} for m in modes}}
    atomic_json(result,args.run/'summary.json')
    from .report import render_markdown
    (args.run/'RESULTS.md').write_text(render_markdown(result))
    print('V7_FULL_EXPERIMENT_PASS '+json.dumps(result['mean_four_severities']),flush=True)
