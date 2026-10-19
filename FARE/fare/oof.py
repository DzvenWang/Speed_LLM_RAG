"""Chronological, auditable STLAformer producer training and paired OOF generation.

Never uses Base_SupportBench val/test for fold training or early stopping.
A fold's producer is trained/validated strictly before its scored window.
"""
from __future__ import annotations
import hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .config import root, cache_dir, backbone_checkpoint, hash_file, dump_json, read_config
from .protocol import SupportDataset

SCHEMA=6

def atomic_torch_save(obj,path):
    """Make epoch checkpoints durable under PBS walltime termination."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    try:
        torch.save(obj,tmp)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def sha_json(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()

def row_keys(meta):
    """Stable row identity including failure, severity and view; *never* unique window_start."""
    cols=['window_start','family','severity','distribution','view_id','view_group_id','protocol_view_id']
    available=[c for c in cols if c in meta.columns]
    if 'window_start' not in available:raise ValueError('Missing window_start')
    keys=[]
    for tup in meta[available].itertuples(index=False,name=None):
        d={k:(v.item() if isinstance(v,np.generic) else v) for k,v in zip(available,tup)}
        d['cache_row']=len(keys)  # preserve even identical repeated views; bound to cache hash
        keys.append(sha_json(d))
    return np.asarray(keys,dtype='U64')

def plan_folds(meta, history=12,horizon=12,nfolds=4,warmup=0.20,purge=12,
               minimum_train=32,minimum_validation=8):
    """Expanding-window folds with *label end* exclusion and time-purged internal validation."""
    starts=meta.window_start.to_numpy(np.int64)
    if len(starts)<int(minimum_train)+int(minimum_validation)+nfolds:raise ValueError('Not enough Train episodes')
    if nfolds<1 or not (0 < warmup < 1):raise ValueError('Invalid folds/warmup')
    uniq=np.unique(starts); begin=int(np.floor(len(uniq)*warmup))
    if begin<4:raise ValueError('Warmup too short')
    groups=np.array_split(uniq[begin:],nfolds)
    folds=[];horizon_offset=history+horizon-1
    for fid,group in enumerate(groups):
        if not len(group):raise ValueError('Empty scoring fold')
        lo,hi=int(group[0]),int(group[-1])
        # Producer's latest training/validation LABEL endpoint must precede first OOF history by purge.
        prior=np.where(starts+horizon_offset+int(purge)<lo)[0]
        prior=prior[np.argsort(starts[prior],kind='stable')]
        if len(prior)<minimum_train+minimum_validation:raise ValueError(f'Fold {fid}: insufficient prior windows ({len(prior)})')
        nval=max(int(minimum_validation),int(round(.1*len(prior))))
        if nval >=len(prior):raise ValueError('No train rows')
        # Time embargo before inner validation too: no model fitting labels overlap early-stop validation history.
        train_last_candidate=int(starts[prior[-nval]]-horizon_offset-int(purge)-1)
        fit=prior[starts[prior]+horizon_offset<=train_last_candidate]
        val=prior[-nval:]
        if len(fit)<minimum_train:raise ValueError(f'Fold {fid}: insufficient purged training prefix')
        score=np.where((starts>=lo)&(starts<=hi))[0]
        # By construction no future sample (nor its label) participates in producer training.
        train_end=int((starts[fit]+horizon_offset).max());val_end=int((starts[val]+horizon_offset).max())
        assert train_end+purge<int(starts[val].min())
        assert val_end+purge<lo
        folds.append({'id':fid,'evaluate_from':lo,'evaluate_until':hi,
                      'fit_rows':fit.tolist(),'val_rows':val.tolist(),'score_rows':score.tolist(),
                      'fit_label_end':train_end,'val_label_end':val_end,'trained_until':val_end,
                      'fit_count':len(fit),'val_count':len(val),'score_count':len(score)})
    all_score=[i for f in folds for i in f['score_rows']]
    if len(set(all_score))!=len(all_score):raise RuntimeError('OOF scoring folds overlap')
    return folds


def save_plan(cache,config,target,dataset,output,nfolds=4,warmup=.2,purge=12):
    ds=SupportDataset(cache,'train')
    keys=row_keys(ds.meta)
    folds=plan_folds(ds.meta,ds.h,ds.p,nfolds,warmup,purge)
    # Use row identities for audit; indices valid only for the exact pinned cache.
    manifest=Path(cache)/'train_manifest.csv'
    identity={'schema_version':SCHEMA,'target':target,'dataset':dataset,'history':ds.h,'horizon':ds.p,
              'cache_path':str(Path(cache).resolve()),'cache_manifest_sha256':hash_file(manifest),
              'train_rows':len(ds),'episode_keys_sha256':sha_json(keys.tolist()),
              'folds':folds,'purge':purge,'warmup':warmup,'nfolds':nfolds,
              'normalization':'fold_prefix_valid_history'}
    identity['plan_sha256']=sha_json({k:v for k,v in identity.items() if k!='plan_sha256'})
    dump_json(output,identity)
    return identity

def load_frozen_plan(path, cache, target, dataset, nfolds, warmup, purge):
    """Read-only plan verification before parallel producer/finalization jobs."""
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Missing frozen OOF plan: {path}; run --plan-only')
    plan=json.loads(path.read_text())
    if sha_json({k:v for k,v in plan.items() if k!='plan_sha256'})!=plan.get('plan_sha256'):
        raise RuntimeError('Corrupted OOF plan digest')
    if (plan['target']!=target or plan['dataset']!=dataset or plan['nfolds']!=nfolds or
        plan['warmup']!=warmup or plan['purge']!=purge or
        Path(plan['cache_path']).resolve()!=Path(cache).resolve()):
        raise RuntimeError('OOF plan flags/cache mismatch')
    ds=SupportDataset(cache,'train')
    if (hash_file(Path(cache)/'train_manifest.csv')!=plan['cache_manifest_sha256'] or
        len(ds)!=plan['train_rows'] or sha_json(row_keys(ds.meta).tolist())!=plan['episode_keys_sha256']):
        raise RuntimeError('OOF plan no longer matches train cache')
    if len(plan['folds'])!=nfolds or [f['id'] for f in plan['folds']]!=list(range(nfolds)):
        raise RuntimeError('Invalid OOF fold indexing')
    return plan


def validate_completed_fold(ckpath, plan, fold, reference_checkpoint, expected_recipe_sha):
    """Only a durable, verified COMPLETED marker makes a producer reusable."""
    ckpath=Path(ckpath);marker=ckpath.parent/'COMPLETED.json'
    if not (ckpath.is_file() and marker.is_file()):return False
    done=json.loads(marker.read_text())
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    proof=ck.get('oof_provenance',{})
    expect={'plan_sha256':plan['plan_sha256'],'fold_id':fold['id'],
            'reference_checkpoint_sha256':hash_file(reference_checkpoint),
            'training_recipe_sha256':expected_recipe_sha}
    if any(done.get(k)!=v or proof.get(k)!=v for k,v in expect.items()):return False
    if done.get('checkpoint_sha256') and done['checkpoint_sha256']!=hash_file(ckpath):return False
    if any(proof.get(k)!=fold[k] for k in ('fit_label_end','val_label_end','evaluate_from','evaluate_until')):return False
    if proof['val_label_end']+int(plan['purge'])>=fold['evaluate_from']:return False
    return True


class ProducerRows(Dataset):
    def __init__(self,ds,indices,reference_mean,reference_std,local_mean,local_std):
        self.ds=ds;self.indices=np.asarray(indices,dtype=np.int64)
        self.refmean=float(reference_mean);self.refstd=float(reference_std)
        self.mean=float(local_mean);self.std=float(local_std)
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        d=self.ds[int(self.indices[i])]
        mask=d['mask'];complete_mask=d['raw_valid']
        xraw=d['x']*self.refstd+self.refmean
        local=torch.where(mask,(xraw-self.mean)/self.std,torch.zeros_like(xraw))
        return {'x':local,'y_raw':d['target_raw'],'future_valid':d['future_valid'],
                'tod_hist':d['tod'],'dow_hist':d['dow'],'index':d['row_id']}

def fold_normalization(ds,fit_rows,refmean,refstd):
    # Prefix only; use genuine raw-valid observations, not artificially masked values.
    arr=ds.arr
    total=0.;sq=0.;count=0
    for low in range(0,len(fit_rows),128):
        ix=np.asarray(fit_rows[low:low+128],dtype=np.int64)
        x=np.asarray(arr['x_full_norm'][ix],dtype=np.float64)*refstd+refmean
        mask=np.asarray(arr['history_observed'][ix]|arr['artificial_missing'][ix],dtype=bool)
        z=x[mask];z=z[np.isfinite(z)]
        total+=float(z.sum());sq+=float(np.dot(z,z));count+=int(len(z))
    if count<2:raise ValueError('No valid raw observations for fold normalization')
    mean=total/count;var=max(sq/count-mean*mean,1e-6)
    return mean,float(np.sqrt(var))


def _external_backbone():
    import sys
    p=root()/'Backbone'
    if not p.is_dir():raise FileNotFoundError(f'Missing Backbone: {p}')
    if str(p) not in sys.path:sys.path.insert(0,str(p))
    from backbone.adapters import build_adapter
    from backbone.model_runner import _optimizer,_scheduler
    from backbone.utils import load_yaml,seed_all
    return build_adapter,_optimizer,_scheduler,load_yaml,seed_all

def train_fold(cfg,plan,fold,reference_checkpoint,output,seed=10,epochs=None,batch_size=None,
               workers=0,device=None):
    """Train actual STLAformer with Backbone adapter/recipe, never a mocked forecast."""
    build_adapter,opt_fn,sched_fn,load_yaml,seed_all=_external_backbone()
    cache=Path(plan['cache_path']);ds=SupportDataset(cache,'train')
    assert hash_file(cache/'train_manifest.csv')==plan['cache_manifest_sha256']
    assert len(ds)==plan['train_rows'] and sha_json(row_keys(ds.meta).tolist())==plan['episode_keys_sha256']
    ref=torch.load(reference_checkpoint,map_location='cpu',weights_only=False)
    if ref.get('model')!='STLAformer' or ref.get('target')!=plan['target'] or ref['num_nodes']!=ds.n:
        raise RuntimeError('Reference checkpoint / cache mismatch')
    mc=load_yaml(root()/'Backbone/configs/model_recipes.yaml')['models']['STLAformer'].copy()
    if epochs is not None:mc['max_epochs']=int(epochs)
    if batch_size is not None:mc['batch_size']=int(batch_size)
    recipe_sha=sha_json(mc)
    seed_all(int(seed)+int(fold['id']))
    mean,std=fold_normalization(ds,fold['fit_rows'],float(ref['mean']),float(ref['std']))
    dev=device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train=ProducerRows(ds,fold['fit_rows'],ref['mean'],ref['std'],mean,std)
    val=ProducerRows(ds,fold['val_rows'],ref['mean'],ref['std'],mean,std)
    train_dl=DataLoader(train,batch_size=mc['batch_size'],shuffle=True,num_workers=workers,
                        pin_memory=dev.type=='cuda',persistent_workers=workers>0,
                        **({'prefetch_factor':2} if workers>0 else {}))
    val_dl=DataLoader(val,batch_size=mc['batch_size'],shuffle=False,num_workers=0)
    adj=np.load(cache/'adjacency.npy')
    adapter=build_adapter('STLAformer',root()/'Backbone/vendor',ds.n,dev,mean,std,mc,adj)
    model=adapter.model;optimizer=opt_fn(model,mc);scheduler,plateau=sched_fn(optimizer,mc)
    from backbone.runner import _loss
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    best=float('inf');best_epoch=0;patience=int(mc['patience']);stale=0
    history=[];updates=0;start_epoch=1
    last=out/'last.pt'
    if last.is_file():
        prev=torch.load(last,map_location='cpu',weights_only=False)
        proof=prev.get('oof_resume_provenance',{})
        if (proof.get('plan_sha256')!=plan['plan_sha256'] or proof.get('fold_id')!=fold['id']
            or proof.get('reference_checkpoint_sha256')!=hash_file(reference_checkpoint)
            or proof.get('training_recipe_sha256')!=recipe_sha):
            raise RuntimeError('Cannot resume fold with incompatible plan or reference checkpoint')
        model.load_state_dict(prev['state_dict'],strict=True)
        optimizer.load_state_dict(prev['optimizer'])
        if scheduler is not None and prev['scheduler'] is not None:scheduler.load_state_dict(prev['scheduler'])
        best=float(prev['best']);best_epoch=int(prev['best_epoch']);stale=int(prev['stale'])
        updates=int(prev['updates']);history=list(prev['history']);start_epoch=int(prev['epoch'])+1
        torch.set_rng_state(prev['rng_state'])
        if torch.cuda.is_available() and prev.get('cuda_rng_states') is not None:
            torch.cuda.set_rng_state_all(prev['cuda_rng_states'])
        print(f'FARE_OOF_EPOCH_RESUME target={plan["target"]} fold={fold["id"]} epoch={start_epoch}',flush=True)
    for epoch in range(start_epoch,int(mc['max_epochs'])+1):
        epoch_start=time.perf_counter()
        if dev.type=='cuda':torch.cuda.reset_peak_memory_stats(dev)
        model.train();losses=[]
        for b in train_dl:
            optimizer.zero_grad(set_to_none=True)
            pred=adapter.train_forward(b,updates)
            loss=_loss(adapter.raw(pred),b['y_raw'].to(dev).float(),b['future_valid'].to(dev),mc.get('train_loss','mae'))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite fold train loss')
            loss.backward()
            if float(mc.get('grad_clip',0))>0:torch.nn.utils.clip_grad_norm_(model.parameters(),float(mc['grad_clip']))
            optimizer.step();losses.append(float(loss.detach().cpu()));updates+=1
        train_seconds=time.perf_counter()-epoch_start
        model.eval();s=0.;n=0
        with torch.no_grad():
            for b in val_dl:
                pred=adapter.eval_forward(b)
                diff=(adapter.raw(pred)-b['y_raw'].to(dev)).abs()
                mask=b['future_valid'].to(dev,dtype=torch.bool)
                s+=float(diff[mask].sum().cpu());n+=int(mask.sum().item())
        metric=s/max(1,n)
        if scheduler is not None:scheduler.step(metric) if plateau else scheduler.step()
        elapsed=time.perf_counter()-epoch_start
        peak_gb=(torch.cuda.max_memory_allocated(dev)/(1024**3)) if dev.type=='cuda' else 0.
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'val_MAE':metric,
                        'elapsed_seconds':round(elapsed,2),'train_seconds':round(train_seconds,2),
                        'max_cuda_allocated_gb':round(peak_gb,3)})
        print(f'FARE_OOF_FOLD_EPOCH target={plan["target"]} fold={fold["id"]} epoch={epoch} MAE={metric:.6f} elapsed_s={elapsed:.1f} train_s={train_seconds:.1f} peak_allocated_gb={peak_gb:.3f} batch_size={mc["batch_size"]} workers={workers}',flush=True)
        if metric<best-1e-8:
            best=metric;best_epoch=epoch;stale=0
            ck={'project_version':'FARE5.0.3_oof','stage':'oof_producer','model':'STLAformer',
                'target':plan['target'],'train_data_mode':'incomplete','state_dict':model.state_dict(),
                'model_config':mc,'num_nodes':ds.n,'mean':mean,'std':std,'seed':int(seed)+int(fold['id']),
                'best_val_MAE':best,'best_epoch':epoch,
                'oof_provenance':{'schema_version':SCHEMA,'plan_sha256':plan['plan_sha256'],
                                  'fold_id':fold['id'],'fit_label_end':fold['fit_label_end'],
                                  'val_label_end':fold['val_label_end'],'evaluate_from':fold['evaluate_from'],
                                  'evaluate_until':fold['evaluate_until'],'purge':plan['purge'],
                                  'fit_count':fold['fit_count'],'val_count':fold['val_count'],
                                  'cache_manifest_sha256':plan['cache_manifest_sha256'],
                                  'reference_checkpoint_sha256':hash_file(reference_checkpoint),
                                  'training_recipe_sha256':recipe_sha,'training_max_epochs':int(mc['max_epochs']),
                                  'training_batch_size':int(mc['batch_size']),
                                  'normalization':'fold_prefix_valid_history','reference_mean':float(ref['mean']),
                                  'reference_std':float(ref['std'])}}
            atomic_torch_save(ck,out/'best.pt')
        else:
            stale+=1
            if stale>=patience:break
        atomic_torch_save({'oof_resume_provenance':{'plan_sha256':plan['plan_sha256'],'fold_id':fold['id'],
                      'reference_checkpoint_sha256':hash_file(reference_checkpoint),
                      'training_recipe_sha256':recipe_sha},
                    'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict() if scheduler is not None else None,
                    'best':best,'best_epoch':best_epoch,'stale':stale,'updates':updates,
                    'history':history,'rng_state':torch.get_rng_state(),
                    'cuda_rng_states':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},last)
    if best_epoch==0 or not (out/'best.pt').is_file():raise RuntimeError('Producer training ended without best.pt')
    dump_json(out/'train_history.json',history)
    dump_json(out/'COMPLETED.json',{'checkpoint_sha256':hash_file(out/'best.pt'),'plan_sha256':plan['plan_sha256'],'fold_id':fold['id'],
                                   'reference_checkpoint_sha256':hash_file(reference_checkpoint),
                                   'training_recipe_sha256':recipe_sha,
                                   'best_epoch':best_epoch,'epochs_completed':history[-1]['epoch']})
    print(f'FARE_OOF_PRODUCER_PASS target={plan["target"]} fold={fold["id"]} checkpoint={out/"best.pt"}',flush=True)
    return out/'best.pt'

@torch.no_grad()
def predict_fold(plan,fold,ckpath,batch_size=16,device=None):
    from .backbone import load_stla
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    proof=ck.get('oof_provenance')
    if not proof:raise RuntimeError('Checkpoint lacks independently saved OOF training provenance')
    for k,expect in [('plan_sha256',plan['plan_sha256']),('fold_id',fold['id']),
                     ('fit_label_end',fold['fit_label_end']),('val_label_end',fold['val_label_end']),
                     ('cache_manifest_sha256',plan['cache_manifest_sha256'])]:
        if proof.get(k)!=expect:raise RuntimeError(f'OOF checkpoint provenance mismatch {k}')
    if max(proof['fit_label_end'],proof['val_label_end'])+int(plan['purge'])>=fold['evaluate_from']:
        raise RuntimeError('OOF producer temporal leakage')
    ds=SupportDataset(plan['cache_path'],'train')
    ids=np.asarray(fold['score_rows'],dtype=np.int64)
    dev=device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=load_stla(ckpath,dev);model.eval()
    pred=[];full=[]
    for pos in range(0,len(ids),batch_size):
        ix=ids[pos:pos+batch_size]
        def t(k,dtype):return torch.as_tensor(np.array(ds.arr[k][ix],copy=True),device=dev,dtype=dtype)
        obs=t('history_observed',torch.bool);miss=t('artificial_missing',torch.bool)
        tod=t('tod_hist',torch.long);dow=t('dow_hist',torch.long)
        x=t('x_obs_norm',torch.float32);xc=t('x_full_norm',torch.float32)
        # Normalization uses only the historical prefix on which this fold was trained.
        refmean,refstd=proof['reference_mean'],proof['reference_std']
        x=torch.where(obs,(x*refstd+refmean-ck['mean'])/ck['std'],torch.zeros_like(x))
        rawvalid=obs|miss
        xc=torch.where(rawvalid,(xc*refstd+refmean-ck['mean'])/ck['std'],torch.zeros_like(xc))
        o=model(x,obs,tod,dow)['prediction'];c=model(xc,rawvalid,tod,dow)['prediction']
        # Return to BASE_SHARED normalization for FARE's y_norm/residual target.
        pred.append(((o*ck['std']+ck['mean']-refmean)/refstd).cpu().numpy())
        full.append(((c*ck['std']+ck['mean']-refmean)/refstd).cpu().numpy())
    if len(pred)==0:raise RuntimeError('No scoring rows')
    return ids,np.concatenate(pred),np.concatenate(full),ck

def export_paired(plan,checkpoints,output,batch_size=16,device=None):
    """One prediction per *row identity*, retaining any repeated window_start."""
    ds=SupportDataset(plan['cache_path'],'train');keys=row_keys(ds.meta)
    records=[];producers=[]
    for fold in plan['folds']:
        ckpath=Path(checkpoints[int(fold['id'])]);ix,pred,complete,ck=predict_fold(plan,fold,ckpath,batch_size,device)
        for j,row in enumerate(ix):records.append((int(row),pred[j],complete[j],fold['val_label_end'],int(fold['id'])))
        producers.append({'fold_id':fold['id'],'checkpoint':str(ckpath.resolve()),
                          'checkpoint_sha256':hash_file(ckpath),'training_provenance':ck['oof_provenance']})
    records.sort(key=lambda r:r[0]);indices=np.array([r[0] for r in records],dtype=np.int64)
    if len(set(indices.tolist()))!=len(indices):raise RuntimeError('Duplicate scored row')
    if len(records)!=sum(f['score_count'] for f in plan['folds']):raise RuntimeError('Incomplete OOF coverage')
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,row_id=indices,episode_key=keys[indices],window_start=ds.meta.window_start.to_numpy(np.int64)[indices],
                        trained_until=np.asarray([r[3] for r in records],dtype=np.int64),
                        fold_id=np.asarray([r[4] for r in records],dtype=np.int16),
                        prediction=np.stack([r[1] for r in records]).astype('float32'),
                        prediction_complete=np.stack([r[2] for r in records]).astype('float32'))
    provenance={'schema_version':SCHEMA,'source_plan_sha256':plan['plan_sha256'],'target':plan['target'],
                'dataset':plan['dataset'],'cache_manifest_sha256':plan['cache_manifest_sha256'],
                'normalization':'fold_prefix_valid_history','records':len(records),'producers':producers,
                'oof_sha256':hash_file(out),'producer_labels_seen':False}
    dump_json(out.with_suffix('.json'),provenance)
    print(f'FARE_PAIRED_OOF_PASS target={plan["target"]} rows={len(indices)} file={out}',flush=True)
    return provenance


def verify_paired(path,ds,strict=True):
    """Fail-closed row identity and provenanced strict temporal validation."""
    p=Path(path);z=np.load(p,allow_pickle=False)
    required=['row_id','episode_key','window_start','trained_until','fold_id','prediction','prediction_complete']
    if any(x not in z.files for x in required):raise RuntimeError(f'OOF source missing schema-6 fields: {required}')
    ix=np.asarray(z['row_id'],dtype=np.int64)
    if len(set(ix.tolist()))!=len(ix) or np.any(ix<0) or np.any(ix>=len(ds.meta)):
        raise RuntimeError('OOF row_id duplicates/out of range')
    keys=row_keys(ds.meta)
    if not np.array_equal(z['episode_key'].astype(str),keys[ix]) or not np.array_equal(z['window_start'],ds.meta.window_start.to_numpy(np.int64)[ix]):
        raise RuntimeError('OOF episode identity mismatch')
    if np.any(np.asarray(z['trained_until'])>=np.asarray(z['window_start'])):
        raise RuntimeError('OOF producer end at/after scoring history')
    expected=(len(ix),ds.p,ds.n)
    for k in ('prediction','prediction_complete'):
        if z[k].shape!=expected or not np.isfinite(z[k]).all():raise RuntimeError(f'OOS {k} shape/nonfinite')
    meta=p.with_suffix('.json')
    if strict:
        if not meta.is_file():raise RuntimeError('OOF audit provenance missing')
        doc=json.loads(meta.read_text())
        if doc.get('schema_version')!=SCHEMA or doc.get('oof_sha256')!=hash_file(p):
            raise RuntimeError('OOF provenance/hash mismatch')
        if doc.get('records')!=len(ix) or doc.get('normalization')!='fold_prefix_valid_history':
            raise RuntimeError('OOF source normalization/row count mismatch')
        if doc.get('cache_manifest_sha256')!=hash_file(ds.path/'train_manifest.csv'):
            raise RuntimeError('OOF source train cache mismatch')
        for item in doc.get('producers',[]):
            ckpath=Path(item['checkpoint'])
            if not ckpath.is_file() or hash_file(ckpath)!=item['checkpoint_sha256']:
                raise RuntimeError('OOF source producer checkpoint missing/changed')
        if not doc.get('producers'):raise RuntimeError('No verified OOF producers')
        seen_folds=set()
        for item in doc['producers']:
            proof=item['training_provenance']; fid=int(item['fold_id'])
            if fid in seen_folds or proof.get('fold_id')!=fid or proof.get('plan_sha256')!=doc['source_plan_sha256']:
                raise RuntimeError('Invalid OOF fold provenance')
            seen_folds.add(fid)
            mask=np.asarray(z['fold_id'])==fid
            if not mask.any() or np.any(np.asarray(z['trained_until'])[mask]!=proof['val_label_end']):
                raise RuntimeError('OOF cutoff mismatch')
            if proof['val_label_end']+int(proof['purge'])>=np.asarray(z['window_start'])[mask].min():
                raise RuntimeError('OOF producer validation embargo violation')
            if proof['fit_label_end']+int(proof['purge'])>=proof.get('evaluate_from',-1):
                raise RuntimeError('OOF producer train embargo violation')
        if set(np.unique(z['fold_id']).astype(int))!=seen_folds:
            raise RuntimeError('OOF fold coverage mismatch')
    return z
