"""Chronological four-severity Selection / Locked validation and ReShift diagnostics.

Besides MAE/RMSE per output this reports, per severity:
  * cf_diagnostics (v0.2.3): harm fractions, dose levels, consistency bins
  * per_family: MAE of backbone/normal/final per failure family (if known)
  * arms: selection fraction, realised gain and harm fraction per arm kind
  * retrieval_harm_bins: harm of the full-dose CF candidate and of the applied
    correction, binned by the neighbours' leave-one-out harm rate
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from reshift.util import to_device
from reshift.training.losses import compute_metrics
from reshift.data.dataset import FAMILIES


def validation_partitions(ds,limit=0):
    starts=np.asarray(ds.window_starts,dtype=np.int64)
    unique=np.unique(starts)
    if len(unique)<20:raise RuntimeError('Need >=20 distinct validation windows for disjoint Selection/Lock')
    sel_cut=unique[max(0,int(len(unique)*.60)-1)]
    lock_cut=unique[min(len(unique)-1,int(len(unique)*.80))]
    selection=np.flatnonzero(starts<=sel_cut)
    locked=np.flatnonzero(starts>=max(int(lock_cut),int(sel_cut)+24))
    if not len(selection) or not len(locked):raise RuntimeError('Empty Selection / locked split')
    if starts[selection].max()+24>starts[locked].min():raise RuntimeError('Overlapping validation targets')
    if limit:
        selection=selection[np.linspace(0,len(selection)-1,min(int(limit),len(selection)),dtype=np.int64)]
        locked=locked[np.linspace(0,len(locked)-1,min(int(limit),len(locked)),dtype=np.int64)]
    return selection.tolist(),locked.tolist()


HARM_EDGES=(-1e9,0.35,0.5,0.65,1e9)
CONS_EDGES=(-1e9,0.0,0.25,0.5,1e9)


@torch.no_grad()
def evaluate(model,datasets,device,mean,std,batch_size=32,indices=None,workers=0,mode='full',
             provider=None,split='val',evidence_mode='full',arm_mask=None):
    model.eval()
    assert mode in ('cf','residual','joint','control','normal','anchor_cf','anchor_ret','anchor_control','full')
    enable_cf=(mode!='normal')
    retrieval_on=(getattr(model,'uses_retrieval',None) is not None and model.uses_retrieval(enable_cf))
    need_ev=(retrieval_on and model.retrieval.needs_evidence and evidence_mode!='none')
    if need_ev and provider is None:raise RuntimeError('evaluation needs a retrieval EvidenceProvider')
    totals={}
    for sev,ds in datasets.items():
        dl=DataLoader(Subset(ds,indices) if indices is not None else ds,batch_size=batch_size,
                      num_workers=workers,shuffle=False,pin_memory=(device.type=='cuda'))
        keys=(('backbone','normal','normal_cf_candidate','final')
              if model.prediction_mode=='normal_anchored' else ('backbone','cf','cf_residual','final'))
        acc={k:[0.,0.,0.] for k in keys}
        cfdiag={'n':0.,'harm':0.,'raw_harm':0.,'gain':0.,'gate':0.,'pred_gain':0.}
        dose_levels=tuple(model.cf.dose_levels) if getattr(model,'dose_enabled',False) else ()
        dosediag={'n':0.,'sum':0.,'reject':0.,'levels':{v:0. for v in dose_levels}}
        consbins=[{'n':0.,'harm':0.,'gain':0.} for _ in range(len(CONS_EDGES)-1)]
        harmbins=[{'n':0.,'cand_harm':0.,'applied_harm':0.,'applied':0.} for _ in range(len(HARM_EDGES)-1)]
        fam={f:{k:[0.,0.] for k in ('backbone','normal','final')} for f in FAMILIES}
        arms=None;seen_dose=seen_cons=seen_harm=False
        for raw in dl:
            b=to_device(raw,device)
            ev=(provider.batch(f'{split}_s{sev}',b['index'],evidence_mode) if need_ev else None)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=='cuda')):
                if model.prediction_mode=='normal_anchored':
                    out=model({k:b[k] for k in ('x','mask','tod','dow')},enable_cf=enable_cf,
                              enable_controller=(mode in ('anchor_control','full')),
                              evidence=ev,arm_mask=arm_mask)
                else:
                    out=model({k:b[k] for k in ('x','mask','tod','dow')},enable_cf=True,
                              enable_residual=(mode!='cf'),enable_controller=(mode in ('control','full')))
            result=compute_metrics(out,b,mean,std)
            valid=b['future_valid'].float();y=b['y_norm'].float()
            if model.prediction_mode=='normal_anchored':
                anchor=out['normal_prediction'].float();candidate_pred=out['normal_cf_candidate'].float()
            else:
                anchor=out['backbone'].float();candidate_pred=out['cf_raw_prediction'].float()
            corrected=out['prediction'].float()
            actual=(anchor-y).abs()-(corrected-y).abs()
            candidate=(anchor-y).abs()-(candidate_pred-y).abs()
            cfdiag['n']+=float(valid.sum())
            cfdiag['harm']+=float(((actual<0).float()*valid).sum())
            cfdiag['raw_harm']+=float(((candidate<0).float()*valid).sum())
            cfdiag['gain']+=float((actual*valid).sum())
            cfdiag['gate']+=float((out['cf_accept_gate'].float()*valid).sum())
            if out['cf_teacher_gain'] is not None:
                cfdiag['pred_gain']+=float((out['cf_teacher_gain'].float()*valid).sum())
            level=out.get('dose_selected_level')
            if level is not None:
                seen_dose=True;level=level.float()
                dosediag['n']+=float(valid.sum());dosediag['sum']+=float((level*valid).sum())
                dosediag['reject']+=float(((level==0).float()*valid).sum())
                for dv in dose_levels:
                    dosediag['levels'][dv]+=float((((level-dv).abs()<1e-6).float()*valid).sum())
            consistency=out.get('consistency')
            if consistency is not None:
                seen_cons=True;contraction=consistency[...,2].float()
                for bi in range(len(consbins)):
                    sel=((contraction>=CONS_EDGES[bi])&(contraction<CONS_EDGES[bi+1])).float()*valid
                    consbins[bi]['n']+=float(sel.sum())
                    consbins[bi]['harm']+=float(((candidate<0).float()*sel).sum())
                    consbins[bi]['gain']+=float((candidate*sel).sum())
            feats=out.get('retrieval_features')
            if feats is not None and ev is not None:
                seen_harm=True;hr=feats[...,5].float()
                applied=(out['arm_selected']>=0).float() if out.get('arm_selected') is not None else torch.ones_like(valid)
                for bi in range(len(harmbins)):
                    sel=((hr>=HARM_EDGES[bi])&(hr<HARM_EDGES[bi+1])).float()*valid
                    harmbins[bi]['n']+=float(sel.sum())
                    harmbins[bi]['cand_harm']+=float(((candidate<0).float()*sel).sum())
                    harmbins[bi]['applied']+=float((applied*sel).sum())
                    harmbins[bi]['applied_harm']+=float(((actual<0).float()*applied*sel).sum())
            if out.get('arm_selected') is not None:
                kinds=out['arm_kinds'];sel_arm=out['arm_selected']
                if arms is None:
                    arms={k:{'n':0.,'gain':0.,'harm':0.} for k in dict.fromkeys(kinds)}
                    arms['reject']={'n':0.,'gain':0.,'harm':0.}
                for j,k in enumerate(kinds):
                    m=(sel_arm==j).float()*valid
                    arms[k]['n']+=float(m.sum());arms[k]['gain']+=float((actual*m).sum())
                    arms[k]['harm']+=float(((actual<0).float()*m).sum())
                arms['reject']['n']+=float(((sel_arm<0).float()*valid).sum())
            if 'family_id' in b:
                fid=b['family_id']
                preds={'backbone':out['backbone'],'normal':out['normal_prediction'] if 'normal_prediction' in out else out['backbone'],
                       'final':out['prediction']}
                err={k:((v.float()*std+mean)-b['y_raw'].float()).abs()*valid for k,v in preds.items()}
                for fi,f in enumerate(FAMILIES):
                    m=(fid==fi).float().view(-1,1,1)
                    if float(m.sum())==0:continue
                    cnt=float((valid*m).sum())
                    for k in fam[f]:
                        fam[f][k][0]+=float((err[k]*m).sum());fam[f][k][1]+=cnt
            for k,vals in result.items():
                for j,v in enumerate(vals):acc[k][j]+=float(v)
        totals[str(sev)]={k:{'mae':a/max(c,1),'rmse':(q/max(c,1))**.5,'n_elements':int(c)}
                          for k,(a,q,c) in acc.items()}
        cd=max(cfdiag['n'],1.)
        diag={'applied_harm_fraction':cfdiag['harm']/cd,'candidate_harm_fraction':cfdiag['raw_harm']/cd,
              'applied_gain_norm_mean':cfdiag['gain']/cd,'accept_gate_mean':cfdiag['gate']/cd,
              'predicted_reference_gain_norm_mean':cfdiag['pred_gain']/cd,'n_elements':int(cfdiag['n']),
              'anchor_kind':('normal' if model.prediction_mode=='normal_anchored' else 'backbone')}
        if seen_dose:
            dn=max(dosediag['n'],1.)
            diag['dose']={'mean_applied_level':dosediag['sum']/dn,'reject_fraction':dosediag['reject']/dn,
                          'level_fraction':{str(v):c/dn for v,c in dosediag['levels'].items()}}
        if seen_cons:
            diag['consistency_bins']=[{'contraction_range':[max(CONS_EDGES[i],-1e3),min(CONS_EDGES[i+1],1e3)],
                                       'fraction':x['n']/cd,'candidate_harm_fraction':x['harm']/max(x['n'],1.),
                                       'candidate_gain_norm_mean':x['gain']/max(x['n'],1.)}
                                      for i,x in enumerate(consbins)]
        if seen_harm:
            diag['retrieval_harm_bins']=[{'loo_harm_range':[max(HARM_EDGES[i],-1e3),min(HARM_EDGES[i+1],1e3)],
                                          'fraction':x['n']/cd,
                                          'cf_candidate_harm_fraction':x['cand_harm']/max(x['n'],1.),
                                          'applied_fraction':x['applied']/max(x['n'],1.),
                                          'applied_harm_fraction':x['applied_harm']/max(x['applied'],1.)}
                                         for i,x in enumerate(harmbins)]
        if arms is not None:
            diag['arms']={k:{'fraction':v['n']/cd,'gain_norm_mean':v['gain']/max(v['n'],1.),
                             'harm_fraction':v['harm']/max(v['n'],1.)} for k,v in arms.items()}
        totals[str(sev)]['cf_diagnostics']=diag
        per_family={f:{k:{'mae':s/max(c,1.),'n_elements':int(c)} for k,(s,c) in d.items()}
                    for f,d in fam.items() if d['final'][1]>0}
        if per_family:totals[str(sev)]['per_family']=per_family
    macro={name:sum(totals[s][name]['mae'] for s in totals)/len(totals) for name in keys}
    return {'per_severity':totals,'macro_mae':macro,'selection_metric':macro['final']}
