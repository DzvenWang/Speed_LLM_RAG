from __future__ import annotations
import warnings
import numpy as np


def softmax_weights(scores, temperature=0.15):
    s = np.asarray(scores, np.float64)
    if len(s) == 0: return np.asarray([], np.float32)
    if not np.all(np.isfinite(s)) or np.nanmax(s) - np.nanmin(s) < 1e-8:
        return np.full(len(s), 1.0 / len(s), np.float32)
    z = (s - np.nanmax(s)) / max(float(temperature), 1e-6)
    e = np.exp(np.clip(z, -40, 40)); e = e / max(float(e.sum()), 1e-12)
    return e.astype(np.float32)


def node_shift_alignment(qx, qobs, cx, cobs, min_node_overlap_steps=2):
    qx=np.asarray(qx,np.float32); cx=np.asarray(cx,np.float32)
    qobs=np.asarray(qobs,bool); cobs=np.asarray(cobs,bool)
    overlap=qobs & cobs & np.isfinite(qx) & np.isfinite(cx)
    diff=qx-cx
    global_shift=float(np.median(diff[overlap])) if np.any(overlap) else 0.0
    counts=overlap.sum(axis=0)
    shifts=np.full(qx.shape[1],global_shift,np.float32)
    valid_nodes=counts>=int(min_node_overlap_steps)
    if np.any(valid_nodes):
        # History is only 12 steps, so vectorized nanmedian is much faster than
        # a Python loop over hundreds of sensors for every candidate.
        tmp=np.where(overlap,diff,np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            med=np.nanmedian(tmp,axis=0)
        good=valid_nodes & np.isfinite(med)
        shifts[good]=med[good].astype(np.float32)
    return shifts


def build_completion(query, rows, retrieved, cfg):
    qx=np.asarray(query['x_obs'].numpy(),np.float32)
    qobs=np.asarray(query['history_observed'].numpy(),bool)
    artificial=np.asarray(query['artificial_missing'].numpy(),bool)
    target = artificial if cfg.get('target','artificial_missing_only')=='artificial_missing_only' else ~qobs
    K=len(retrieved['local']); w=softmax_weights(retrieved['score'], cfg.get('score_temperature',0.15))
    vals=[]; avail=[]; shifts=[]
    for j,li in enumerate(retrieved['local']):
        r=rows[int(li)]
        sh=node_shift_alignment(qx,qobs,r.x_obs,r.history_observed,cfg['min_node_overlap_steps']); shifts.append(sh)
        aligned=np.asarray(r.x_obs,np.float32)+sh[None,:]
        a=np.asarray(r.history_observed,bool) & target & np.isfinite(aligned)
        vals.append(aligned); avail.append(a)
    if K==0:
        return {'x_aug':qx.copy(),'fill_mask':np.zeros_like(target),'entry_confidence':np.zeros_like(qx,np.float32),
                'node_confidence':np.zeros(qx.shape[1],np.float32),'coverage':0.0,'weights':w,'shifts':[],
                'candidate_values':np.zeros((0,)+qx.shape,np.float32),'candidate_available':np.zeros((0,)+target.shape,bool)}
    vals=np.stack(vals); avail=np.stack(avail)
    num=np.sum(w[:,None,None]*np.where(avail,vals,0.0),axis=0)
    den=np.sum(w[:,None,None]*avail.astype(np.float32),axis=0)
    mean=np.divide(num,np.maximum(den,1e-8))
    var_num=np.sum(w[:,None,None]*np.where(avail,(vals-mean[None,:,:])**2,0.0),axis=0)
    var=np.divide(var_num,np.maximum(den,1e-8)); std=np.sqrt(np.maximum(var,0.0))
    support_count=avail.sum(axis=0)
    agreement=np.exp(-std/max(float(cfg['consensus_scale_norm']),1e-6))
    conf=den*agreement
    fill=target & (support_count>=int(cfg['min_support_cases_per_entry'])) & (den>=float(cfg['min_weighted_coverage']))
    x_aug=qx.copy(); x_aug[fill]=mean[fill]
    node_conf=np.zeros(qx.shape[1],np.float32)
    for n in range(qx.shape[1]):
        m=target[:,n]
        if np.any(m): node_conf[n]=float(np.mean(conf[m]))
    coverage=float(fill.sum())/max(int(target.sum()),1)
    return {'x_aug':x_aug.astype(np.float32),'fill_mask':fill,'entry_confidence':conf.astype(np.float32),
            'node_confidence':np.clip(node_conf,0,1),'coverage':coverage,'weights':w,'shifts':shifts,
            'candidate_values':vals.astype(np.float32),'candidate_available':avail.astype(bool)}


def build_transition(query, rows, retrieved, completion, std, cfg):
    K=len(retrieved['local']); N=int(np.asarray(query['x_obs'].numpy()).shape[1]); H=int(np.asarray(rows[int(retrieved['local'][0])].y_raw).shape[0]) if K else 0
    w=softmax_weights(retrieved['score'], cfg.get('score_temperature',0.15))
    vals=[]; avail=[]
    shifts=completion.get('shifts',[])
    for j,li in enumerate(retrieved['local']):
        r=rows[int(li)]
        sh = shifts[j] if j < len(shifts) else np.zeros(N,np.float32)
        aligned=np.asarray(r.y_raw,np.float32)+float(std)*sh[None,:]
        a=np.asarray(r.future_valid,bool)&np.isfinite(aligned)
        vals.append(aligned); avail.append(a)
    if K==0:
        return {'forecast':np.zeros((H,N),np.float32),'valid':np.zeros((H,N),bool),'confidence':np.zeros((H,N),np.float32),
                'weights':w,'candidate_forecasts':[]}
    vals=np.stack(vals); avail=np.stack(avail)
    num=np.sum(w[:,None,None]*np.where(avail,vals,0.0),axis=0)
    den=np.sum(w[:,None,None]*avail.astype(np.float32),axis=0)
    mean=np.divide(num,np.maximum(den,1e-8))
    var_num=np.sum(w[:,None,None]*np.where(avail,(vals-mean[None,:,:])**2,0.0),axis=0)
    var=np.divide(var_num,np.maximum(den,1e-8)); sd=np.sqrt(np.maximum(var,0.0))
    count=avail.sum(axis=0)
    agreement=np.exp(-sd/max(float(cfg['consensus_scale_raw']),1e-6))
    conf=den*agreement
    valid=(count>=int(cfg['min_support_cases_per_entry']))&(den>=float(cfg['min_weighted_coverage']))
    candidates=[]
    for j in range(K):
        c=vals[j].copy(); c[~avail[j]]=np.nan; candidates.append(c.astype(np.float32))
    return {'forecast':mean.astype(np.float32),'valid':valid,'confidence':np.clip(conf,0,1).astype(np.float32),
            'weights':w,'candidate_forecasts':candidates}


def fusion_scope(query, mode, horizon):
    d=np.asarray(query['direct_mask'].numpy(),bool); s=np.asarray(query['support_mask'].numpy(),bool)
    H=int(horizon)
    def broad(m):
        return np.broadcast_to(m[None,:],(H,len(m))).copy() if m.ndim==1 else m.copy()
    if mode=='direct_only': return broad(d)
    if mode=='support_only': return broad(s)
    if mode=='all_valid': return np.ones((H,len(d)),dtype=bool)
    return broad(d|s)


def fuse_predictions(slow_pred, completion_pred, transition, completion, query, ccfg, tcfg, fcfg, use_completion=True, use_transition=True, completion_multiplier=1.0):
    slow=np.asarray(slow_pred,np.float32); H,N=slow.shape
    scope=fusion_scope(query,fcfg.get('scope','direct_and_support'),H)
    gc=np.zeros((H,N),np.float32); ge=np.zeros((H,N),np.float32)
    if use_completion and completion_pred is not None:
        nc=np.asarray(completion['node_confidence'],np.float32)
        gc=np.broadcast_to(nc[None,:],(H,N)).copy()*float(ccfg['max_gate'])*float(np.clip(completion_multiplier,0.0,1.0))
    if use_transition and transition is not None:
        ge=np.asarray(transition['confidence'],np.float32)*float(tcfg['max_gate'])
        ge[~np.asarray(transition['valid'],bool)]=0.0
    gc[~scope]=0.0; ge[~scope]=0.0
    cap=float(fcfg['max_external_weight']); total=gc+ge
    scale=np.ones_like(total); m=total>cap; scale[m]=cap/np.maximum(total[m],1e-8)
    gc*=scale; ge*=scale
    final=slow.copy()
    if completion_pred is not None:
        final += gc*(np.asarray(completion_pred,np.float32)-slow)
    if transition is not None:
        final += ge*(np.asarray(transition['forecast'],np.float32)-slow)
    return final.astype(np.float32),gc,ge


def build_nodewise_completion(query, rows, node_retrieved, cfg):
    """Compose missing history node-by-node from independently retrieved supports.

    Each unique historical case is aligned only once and cached. Different missing
    nodes can then use different subsets of these aligned cases without multiplying
    the expensive alignment cost by the number of sensors.
    """
    qx=np.asarray(query['x_obs'].numpy(),np.float32)
    qobs=np.asarray(query['history_observed'].numpy(),bool)
    artificial=np.asarray(query['artificial_missing'].numpy(),bool)
    target=artificial if cfg.get('target','artificial_missing_only')=='artificial_missing_only' else ~qobs
    x_aug=qx.copy(); conf=np.zeros_like(qx,np.float32); fill=np.zeros_like(target,bool)
    candidate_values_by_node={}; candidate_available_by_node={}; support_meta=[]
    if node_retrieved is None:
        return {'x_aug':x_aug,'fill_mask':fill,'entry_confidence':conf,'node_confidence':np.zeros(qx.shape[1],np.float32),
                'coverage':0.0,'node_supports':{},'candidate_values_by_node':candidate_values_by_node,
                'candidate_available_by_node':candidate_available_by_node,'support_meta':support_meta}

    # Cache aligned full histories once per unique candidate used by any missing node.
    unique_local=sorted({int(li) for sup in node_retrieved.get('nodes',{}).values() for li in np.asarray(sup['local'],np.int64)})
    aligned_cache={}
    for li in unique_local:
        r=rows[li]
        sh=node_shift_alignment(qx,qobs,r.x_obs,r.history_observed,cfg['min_node_overlap_steps'])
        aligned_cache[li]=(np.asarray(r.x_obs,np.float32)+sh[None,:],np.asarray(r.history_observed,bool))

    for n,sup in node_retrieved.get('nodes',{}).items():
        n=int(n); local=np.asarray(sup['local'],np.int64); weights=np.asarray(sup['weights'],np.float32)
        vals=[]; av=[]
        for j,li in enumerate(local):
            aligned,cobs=aligned_cache[int(li)]
            v=aligned[:,n]; a=cobs[:,n] & target[:,n] & np.isfinite(v)
            vals.append(v); av.append(a)
            support_meta.append({'node':n,'rank':j+1,'case_index':int(rows[int(li)].index),'score':float(sup['score'][j]),
                                 'weight':float(weights[j]),'local_observed_match':float(sup['local_observed_match'][j]),
                                 'node_complementarity':float(sup['node_complementarity'][j])})
        if not vals:
            continue
        vals=np.stack(vals).astype(np.float32); av=np.stack(av).astype(bool)
        candidate_values_by_node[n]=vals; candidate_available_by_node[n]=av
        num=np.sum(weights[:,None]*np.where(av,vals,0.0),axis=0)
        den=np.sum(weights[:,None]*av.astype(np.float32),axis=0)
        mean=np.divide(num,np.maximum(den,1e-8))
        var_num=np.sum(weights[:,None]*np.where(av,(vals-mean[None,:])**2,0.0),axis=0)
        var=np.divide(var_num,np.maximum(den,1e-8)); sd=np.sqrt(np.maximum(var,0.0))
        agreement=np.exp(-sd/max(float(cfg['consensus_scale_norm']),1e-6))
        c=den*agreement; count=av.sum(axis=0)
        m=target[:,n] & (count>=int(cfg['min_support_cases_per_entry'])) & (den>=float(cfg['min_weighted_coverage']))
        x_aug[m,n]=mean[m]; conf[m,n]=c[m]; fill[m,n]=True

    x_aug[qobs]=qx[qobs]; fill[qobs]=False; conf[qobs]=0.0
    node_conf=np.zeros(qx.shape[1],np.float32)
    for n in range(qx.shape[1]):
        m=target[:,n]
        if np.any(m): node_conf[n]=float(np.mean(conf[m]))
    coverage=float(fill.sum())/max(int(target.sum()),1)
    return {'x_aug':x_aug.astype(np.float32),'fill_mask':fill,'entry_confidence':np.clip(conf,0,1).astype(np.float32),
            'node_confidence':np.clip(node_conf,0,1),'coverage':coverage,'node_supports':node_retrieved.get('nodes',{}),
            'candidate_values_by_node':candidate_values_by_node,'candidate_available_by_node':candidate_available_by_node,
            'support_meta':support_meta}



def build_patchwise_completion(query, prepared, patch_retrieved, cfg):
    """Compose missing history from independently selected node x temporal-patch supports.

    `prepared` is produced once from the causal global candidate pool, so all selected
    supports use the same visible-data alignment and no hidden current truth.
    """
    qx=np.asarray(query['x_obs'].numpy(),np.float32)
    qobs=np.asarray(query['history_observed'].numpy(),bool)
    artificial=np.asarray(query['artificial_missing'].numpy(),bool)
    target=artificial if cfg.get('target','artificial_missing_only')=='artificial_missing_only' else ~qobs
    x_aug=qx.copy(); conf=np.zeros_like(qx,np.float32); fill=np.zeros_like(target,bool)
    support_meta=[]; patch_supports={}
    if prepared is None or patch_retrieved is None:
        return {'x_aug':x_aug,'fill_mask':fill,'entry_confidence':conf,'node_confidence':np.zeros(qx.shape[1],np.float32),
                'coverage':0.0,'patch_supports':patch_supports,'support_meta':support_meta}
    for key,sup in patch_retrieved.get('patches',{}).items():
        n=int(sup['node']); s=int(sup['start']); e=int(sup['end']); pool_rank=np.asarray(sup['pool_rank'],np.int64); w=np.asarray(sup['weights'],np.float32)
        vals=[]; av=[]
        for j,pr in enumerate(pool_rank):
            v=prepared['aligned'][int(pr),s:e,n]
            a=prepared['cobs'][int(pr),s:e,n] & target[s:e,n] & np.isfinite(v)
            vals.append(v); av.append(a)
            support_meta.append({'node':n,'patch_id':int(sup['patch_id']),'patch_start':s,'patch_end':e,'rank':j+1,
                'case_index':int(sup['case_index'][j]),'score':float(sup['score'][j]),'weight':float(w[j]),'selection_source':sup.get('source','unknown'),
                'p_helpful':float(sup['p_helpful'][j]) if np.isfinite(sup['p_helpful'][j]) else float('nan'),
                'p_harmful':float(sup['p_harmful'][j]) if np.isfinite(sup['p_harmful'][j]) else float('nan'),
                'patch_observed_match':float(sup['patch_observed_match'][j]),'patch_complementarity':float(sup['patch_complementarity'][j])})
        if not vals: continue
        vals=np.stack(vals).astype(np.float32); av=np.stack(av).astype(bool)
        num=np.sum(w[:,None]*np.where(av,vals,0.0),axis=0); den=np.sum(w[:,None]*av.astype(np.float32),axis=0)
        mean=np.divide(num,np.maximum(den,1e-8)); var_num=np.sum(w[:,None]*np.where(av,(vals-mean[None,:])**2,0.0),axis=0)
        var=np.divide(var_num,np.maximum(den,1e-8)); sd=np.sqrt(np.maximum(var,0.0)); agreement=np.exp(-sd/max(float(cfg['consensus_scale_norm']),1e-6))
        c=den*agreement; count=av.sum(axis=0); m=target[s:e,n] & (count>=int(cfg['min_support_cases_per_entry'])) & (den>=float(cfg['min_weighted_coverage']))
        loc=np.arange(s,e); x_aug[loc[m],n]=mean[m]; conf[loc[m],n]=c[m]; fill[loc[m],n]=True
        patch_supports[(n,int(sup['patch_id']))]={'case_index':np.asarray(sup['case_index'],np.int64),'weights':w.copy(),'fill_count':int(m.sum()),'source':sup.get('source','unknown')}
    x_aug[qobs]=qx[qobs]; fill[qobs]=False; conf[qobs]=0.0
    node_conf=np.zeros(qx.shape[1],np.float32)
    for n in range(qx.shape[1]):
        m=target[:,n]
        if np.any(m): node_conf[n]=float(np.mean(conf[m]))
    coverage=float(fill.sum())/max(int(target.sum()),1)
    return {'x_aug':x_aug.astype(np.float32),'fill_mask':fill,'entry_confidence':np.clip(conf,0,1).astype(np.float32),
            'node_confidence':np.clip(node_conf,0,1),'coverage':coverage,'patch_supports':patch_supports,'support_meta':support_meta}
