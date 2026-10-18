"""Real FARE v4.0.3 checkpoint -> one-pass paired diagnostic shards.

No retraining; no synthetic predictions. Uses cached full forward and documented
FARE intervention modes. The oracle is limited to individually isolated members
of the *selected* evidence set, and is never called a search-pool oracle.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from .data import sha256_file

FARE_DIAGNOSTIC_ADAPTER_READY = True

MODE_MAP = {
    'full': 'full',
    'no_evidence': 'no_evidence',
    'hidden_only': 'no_forecast_correction',
    'forecast_only': 'no_hidden_injection',
    'no_gate': 'no_risk_control',
    'similarity_only': 'state_similarity_only',
    'random': 'random_legal_memory',
    'shuffled': 'shuffled_legal_memory',
    'no_support_demand': 'no_support_demand',
    'no_direction': 'no_direction',
    'no_propagation_alignment': 'no_propagation_alignment',
    'no_set_interaction': 'no_set_interaction',
    'no_counterfactual_utility': 'no_counterfactual_utility',
}

FAMILY_ALIASES = ('failure_family','failure_type','mask_family','family','failure_mode','challenge_family')

def _family_column(ds):
    for col in FAMILY_ALIASES:
        if col in ds.meta.columns:
            return col
    raise RuntimeError(f'Cannot identify actual failure family in {ds.split}_manifest.csv. Columns={list(ds.meta.columns)}. '
                       'Refusing to assign synthetic family labels.')

def _indices_for_family(ds,family,limit,seed,severity):
    col=_family_column(ds)
    names=ds.meta.iloc[ds.ix][col].astype(str).to_numpy()
    idx=np.where(names==family)[0]
    if limit and len(idx)>limit:
        hashes=[int(hashlib.sha256(f'{int(ds.ix[i])}|{seed}|{severity}|{family}'.encode()).hexdigest()[:16],16) for i in idx]
        idx=np.sort(idx[np.argsort(np.asarray(hashes,dtype=np.uint64),kind='stable')[:limit]])
    return idx.tolist()

def _one_selected_candidate(model,batch,full,slot):
    sel=full['selection']; prepared=full['prepared']
    w=sel['weights']
    candidate={k:(v.clone() if torch.is_tensor(v) else v) for k,v in sel.items()}
    keep=torch.zeros(w.shape[2],dtype=torch.bool,device=w.device);keep[slot]=True
    candidate['weights'][:,:,~keep]=0
    candidate['selected_valid'][:,:,~keep]=False
    candidate['null']=sel['null']+w[:,:,~keep].sum((2,3))
    candidate['correction']=(candidate['weights']*candidate['selected_values']).sum((2,3))
    return model.predict_with_selection(batch,prepared,candidate)['prediction']

def _cpu_ready(model,memory,ds,seed,checkpoint,split,severity,target,groups,max_queries):
    col=_family_column(ds)
    found=sorted(set(ds.meta.iloc[ds.ix][col].astype(str)))
    missing=[g for g in groups if g not in found]
    if missing:raise RuntimeError(f'{target} s{severity} {split} missing families {missing}; found={found}')
    if not len(ds):raise RuntimeError(f'{target} s{severity} {split} has no queries')
    return {'target':target,'severity':severity,'split':split,'family_column':col,'family_counts':{
        g:len(_indices_for_family(ds,g,max_queries,seed,severity)) for g in groups},
        'memory_provenance':memory.metadata.get('residual_provenance')}

@torch.no_grad()
def export_once(*,target:str,checkpoint:str,output_dir:str,severities:list[int],
                failure_families:list[str],max_queries_per_group:int,seed:int,
                split:str,config:dict):
    from fare.config import root,cache_dir
    from fare.pipeline import prepare
    from fare.protocol import SupportDataset,device_batch,make_loader
    from fare.training import seed_everything
    from fare.diagnostics_once.report import save_json
    if split not in ('val','test'):raise ValueError('Diagnostic split must be val or test')
    if split=='test' and not config.get('allow_exploratory_test',False):
        raise ValueError('Using Test for exploratory architecture audit requires config.allow_exploratory_test=true')
    seed_everything(seed)
    checkpoint=Path(checkpoint).resolve()
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not all(k in ck for k in ('state_dict','config','metadata')):
        raise RuntimeError('Not a complete FARE training checkpoint')
    cfg=ck['config']
    ex=cfg['experiment']
    if (ex['target']!=target or ex['seed']!=seed or ex['dataset']!=config.get('dataset',ex['dataset'])):
        raise RuntimeError('Checkpoint target/seed/dataset does not match one-pass request')
    meta=ck['metadata']
    backbone_path=Path(meta['backbone_checkpoint'])
    memory_path=Path(meta['memory_path'])
    if not backbone_path.is_file():raise FileNotFoundError(f'Original frozen STLA checkpoint missing: {backbone_path}')
    if not (memory_path/'manifest.json').is_file():raise FileNotFoundError(f'FARE memory not found: {memory_path}')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,memory,_,formal_val,_,prepared_meta=prepare(
        cfg,device,backbone_file=str(backbone_path),memory_file=str(memory_path),build=False,cuda_safety=False)
    if (meta['backbone_sha256']!=prepared_meta['backbone_sha256'] or
            meta['memory_sha256']!=prepared_meta['memory_sha256']):
        raise RuntimeError('Original checkpoint differs from current frozen backbone/memory manifest')
    model.load_state_dict(ck['state_dict'],strict=True)
    model.eval()
    model_sha=sha256_file(checkpoint)
    outroot=Path(output_dir).resolve();outroot.mkdir(parents=True,exist_ok=True)
    groups=list(failure_families) if failure_families else ['entry_random','regional_outage','directional_outage','temporal_outage','sensor_outage']
    export_modes={**MODE_MAP}
    report=[]
    for sev in severities:
        path=cache_dir(cfg,'evaluation',sev)
        fn=path/f'{split}_manifest.csv'
        if not fn.exists():
            raise FileNotFoundError(f'Missing {split} manifest for s{sev}: {fn}. '
                'Do not silently substitute test for validation. Choose split=test with explicit exploratory opt-in.')
        ds=SupportDataset(path,split,None,strategy='spread')
        info=_cpu_ready(model,memory,ds,seed,checkpoint,split,sev,target,groups,max_queries_per_group)
        report.append(info)
        print('FARE_DIAG_DATA '+json.dumps(info),flush=True)
        for family in groups:
            select=_indices_for_family(ds,family,max_queries_per_group,seed,sev)
            if not select:raise RuntimeError(f'Empty stratum {target} s{sev} {family}')
            shard=outroot/f'{target}_s{sev:02}_{family}.npz'
            if shard.exists():
                from .data import load_shard
                try:
                    cached=load_shard(shard,max_queries=0,allowed_splits=(split,))
                    if (cached.metadata['checkpoint_sha256']==model_sha and cached.n==len(select) and
                            cached.metadata.get('source_split')==split):
                        print('FARE_DIAG_SHARD_REUSED '+str(shard),flush=True)
                        continue
                except Exception:pass
                shard.unlink()
            subset=Subset(ds,select)
            accum={key:[] for key in export_modes}
            accum.update({'null_evidence':[]})
            ytrue=[];valid=[];ids=[];origins=[];candidates=[]
            extras={key:[] for key in ('alpha','null_weight','demand','mask_ratio','propagation','future_start','memory_last_future_end')}
            # One GPU batch, one full retrieval; downstream modes reuse full's prepared state.
            for raw in make_loader(subset,int(config.get('inference_batch_size',2)),workers=0):
                batch=device_batch(raw,device)
                full=model(batch,memory,purge=cfg['memory']['purge_steps'],return_details=True)
                mean=model.backbone.mean;std=model.backbone.std
                def original(x):return (x.float()*std+mean).detach().cpu().numpy().astype('float32')
                for name,mode in export_modes.items():
                    ans=full if mode=='full' else model.intervene_cached(batch,memory,full,mode,cfg['memory']['purge_steps'])
                    accum[name].append(original(ans['prediction']))
                # A separate explicit null action; equality with no_evidence is audited, not assumed.
                empty=model._empty_selection(full['base_prediction'])
                null_out=model.predict_with_selection(batch,full['prepared'],empty,'no_evidence')
                accum['null_evidence'].append(original(null_out['prediction']))
                sel=full['selection'];k=sel['selected_ids'].shape[2]
                one=[]
                for slot in range(min(k,int(config.get('oracle_selected_slots',8)))):
                    one.append(original(_one_selected_candidate(model,batch,full,slot)))
                if one:candidates.append(np.stack(one,axis=1))
                ytrue.append(raw['target_raw'].numpy())
                valid.append(raw['future_valid'].numpy())
                ids.append(raw['row_id'].numpy())
                origins.append(raw['origin'].numpy())
                extras['alpha'].append(full['alpha'].detach().cpu().numpy())
                extras['null_weight'].append(sel['null'].detach().cpu().numpy())
                extras['demand'].append(full['support']['demand'].detach().cpu().numpy())
                extras['propagation'].append(full['support']['forward_coverage'].detach().cpu().numpy())
                extras['mask_ratio'].append((~raw['mask'].bool()).float().mean((1,2)).numpy())
                extras['future_start'].append(raw['origin'].numpy()+1)
                # Timestamp bound for the actual selected evidence, excluding invalid placeholder slots.
                memory_last=(memory.starts+memory.x.shape[1]+memory.future.shape[1]-1).cpu()
                ids_selected=sel['selected_ids'].detach().cpu().long().clamp(0,len(memory_last)-1)
                chosen_end=memory_last[ids_selected]
                selected_valid=sel['selected_valid'].any((-1,-2)).detach().cpu()
                chosen_end=torch.where(selected_valid,chosen_end,torch.full_like(chosen_end,-1))
                extras['memory_last_future_end'].append(chosen_end.numpy())
            metadata={'target':target,'severity':int(sev),'failure_family':family,'split':split,
                      'source_split':split,'seed':seed,'checkpoint':str(checkpoint),
                      'checkpoint_sha256':model_sha,'metrics_space':'original_units',
                      'dataset':ex['dataset'],'version':'4.0.4','residual_provenance':memory.metadata.get('residual_provenance'),
                      'oracle_scope':'selected_slot_only_not_pool_oracle',
                      'architecture_selection_safe':split=='val'}
            payload={'y_true':np.concatenate(ytrue),'valid_mask':np.concatenate(valid),
                     'query_id':np.concatenate(ids), 'metadata_json':np.array(json.dumps(metadata,sort_keys=True))}
            payload.update({'pred__'+k:np.concatenate(v) for k,v in accum.items()})
            if candidates:payload['pred_candidates']=np.concatenate(candidates)
            payload.update({k:np.concatenate(v) for k,v in extras.items()})
            temp=shard.with_suffix('.tmp.npz')
            np.savez_compressed(temp,**payload)
            temp.replace(shard)
            print(f'FARE_DIAG_EXPORT_COMPLETE {target} s{sev} {family} n={len(select)} file={shard}',flush=True)
    save_json(outroot/f'{target}_source_audit.json',{'checkpoint':str(checkpoint),'checkpoint_sha256':model_sha,
             'source_split':split,'cache_families':report,'modes':list(accum),'oracle_scope':'selected-slot-only',
             'residual_provenance':memory.metadata.get('residual_provenance')})
    return report
