"""Chronological four-severity Selection and non-overlapping Locked validation."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from corsta_v2.util import to_device
from corsta_v2.training.losses import compute_metrics


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
        indices=np.linspace(0,len(selection)-1,min(int(limit),len(selection)),dtype=np.int64)
        selection=selection[indices]
        lockidx=np.linspace(0,len(locked)-1,min(int(limit),len(locked)),dtype=np.int64)
        locked=locked[lockidx]
    return selection.tolist(),locked.tolist()


@torch.no_grad()
def evaluate(model,datasets,device,mean,std,batch_size=32,indices=None,workers=0,mode='full'):
    model.eval()
    assert mode in ('cf','residual','joint','control','normal','anchor_cf','anchor_control','full')
    totals={}
    for sev,ds in datasets.items():
        dl=DataLoader(Subset(ds,indices) if indices is not None else ds,batch_size=batch_size,num_workers=workers,shuffle=False,
                      pin_memory=(device.type=='cuda'))
        keys=(('backbone','normal','normal_cf_candidate','final')
              if model.prediction_mode=='normal_anchored' else
              ('backbone','cf','cf_residual','final'))
        acc={k:[0.,0.,0.] for k in keys}
        cfdiag={'n':0.,'harm':0.,'raw_harm':0.,'gain':0.,'gate':0.,'pred_gain':0.}
        dose_levels=tuple(model.cf.dose_levels) if getattr(model,'dose_enabled',False) else ()
        dosediag={'n':0.,'sum':0.,'reject':0.,'levels':{v:0. for v in dose_levels}}
        # Consistency-contraction bins: how harmful is the FULL-dose candidate
        # as a function of the label-free view-consistency feature (mechanism A).
        cons_edges=(-1e9,0.0,0.25,0.5,1e9)
        consbins=[{'n':0.,'harm':0.,'gain':0.} for _ in range(len(cons_edges)-1)]
        seen_dose=False;seen_cons=False
        for raw in dl:
            b=to_device(raw,device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,
                                enabled=(device.type=='cuda')):
                if model.prediction_mode=='normal_anchored':
                    out=model({k:b[k] for k in ('x','mask','tod','dow')},
                              enable_cf=(mode!='normal'),
                              enable_controller=(mode in ('anchor_control','full')))
                else:
                    out=model({k:b[k] for k in ('x','mask','tod','dow')},
                              enable_cf=True,enable_residual=(mode!='cf'),
                              enable_controller=(mode in ('control','full')))
            result=compute_metrics(out,b,mean,std)
            valid=b['future_valid'].float()
            if model.prediction_mode=='normal_anchored':
                anchor=out['normal_prediction'].float()
                candidate_pred=out['normal_cf_candidate'].float()
                corrected=out['prediction'].float()
            else:
                anchor=out['backbone'].float()
                candidate_pred=out['cf_raw_prediction'].float()
                corrected=out['cf_prediction'].float()
            actual=(anchor-b['y_norm'].float()).abs()-(corrected-b['y_norm'].float()).abs()
            candidate=(anchor-b['y_norm'].float()).abs()-(candidate_pred-b['y_norm'].float()).abs()
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
                dosediag['n']+=float(valid.sum())
                dosediag['sum']+=float((level*valid).sum())
                dosediag['reject']+=float(((level==0).float()*valid).sum())
                for dv in dose_levels:
                    dosediag['levels'][dv]+=float((((level-dv).abs()<1e-6).float()*valid).sum())
            consistency=out.get('consistency')
            if consistency is not None:
                seen_cons=True;contraction=consistency[...,2].float()
                for bi in range(len(consbins)):
                    sel=((contraction>=cons_edges[bi])&(contraction<cons_edges[bi+1])).float()*valid
                    consbins[bi]['n']+=float(sel.sum())
                    consbins[bi]['harm']+=float(((candidate<0).float()*sel).sum())
                    consbins[bi]['gain']+=float((candidate*sel).sum())
            for k,vals in result.items():
                for j,v in enumerate(vals):acc[k][j]+=float(v)
        totals[str(sev)]={k:{'mae':a/max(c,1),'rmse':(q/max(c,1))**.5,'n_elements':int(c)}
                          for k,(a,q,c) in acc.items()}
        cd=max(cfdiag['n'],1.)
        totals[str(sev)]['cf_diagnostics']={
            'applied_harm_fraction':cfdiag['harm']/cd,
            'candidate_harm_fraction':cfdiag['raw_harm']/cd,
            'applied_gain_norm_mean':cfdiag['gain']/cd,
            'accept_gate_mean':cfdiag['gate']/cd,
            'predicted_reference_gain_norm_mean':cfdiag['pred_gain']/cd,
            'n_elements':int(cfdiag['n']),
            'anchor_kind':('normal' if model.prediction_mode=='normal_anchored' else 'backbone')}
        if seen_dose:
            dn=max(dosediag['n'],1.)
            totals[str(sev)]['cf_diagnostics']['dose']={
                'mean_applied_level':dosediag['sum']/dn,
                'reject_fraction':dosediag['reject']/dn,
                'level_fraction':{str(v):c/dn for v,c in dosediag['levels'].items()}}
        if seen_cons:
            totals[str(sev)]['cf_diagnostics']['consistency_bins']=[
                {'contraction_range':[max(cons_edges[i],-1e3),min(cons_edges[i+1],1e3)],
                 'fraction':b['n']/cd,
                 'candidate_harm_fraction':b['harm']/max(b['n'],1.),
                 'candidate_gain_norm_mean':b['gain']/max(b['n'],1.)}
                for i,b in enumerate(consbins)]
    macro={name:sum(totals[sev][name]['mae'] for sev in totals)/len(totals) for name in keys}
    return {'per_severity':totals,'macro_mae':macro,'selection_metric':macro['final']}
