"""Streaming C1/C2/C3 statistics from immutable predictions."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from .metrics import Accumulator,GroupedMetrics,group_masks
from .interventions import fixed_predictions,run_intervention


def arr(value):return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)

class Calibrator:
    def __init__(self):self.stats=defaultdict(lambda:defaultdict(float))
    def update(self,pred_gain,pred_harm,true_gain,true_harm,valid):
        # Gain/risk predictions in normalized space, consistent with C3 training.
        for action in range(2):
            x=pred_gain[...,action];y=true_gain[...,action]
            r=pred_harm[...,action];h=true_harm[...,action]
            for mode,values,bounds in [('gain',x,[-.05,-.01,0,.01,.05]),
                                       ('risk',1/(1+np.exp(-np.clip(r,-30,30))),[.1,.3,.5,.7,.9])]:
                labels=np.digitize(values,bounds)
                for label in np.unique(labels[valid]):
                    mask=valid & (labels==label)
                    if not mask.any():continue
                    a=self.stats[(action+1,mode,int(label))]
                    n=int(mask.sum());a['n']+=n
                    a['predicted_gain_sum']+=float(x[mask].sum())
                    a['actual_gain_sum']+=float(y[mask].sum())
                    a['predicted_harm_sum']+=float((1/(1+np.exp(-np.clip(r[mask],-30,30)))).sum())
                    a['actual_harm_sum']+=int(h[mask].sum())
                    a['gain_abs_error_sum']+=float(np.abs(x[mask]-y[mask]).sum())
    def rows(self):
        out=[]
        for (a,mode,b),v in sorted(self.stats.items()):
            n=v['n'];out.append(dict(action=('local','local_propagation')[a-1],bin_type=mode,bin_index=b,
                valid_positions=int(n),predicted_gain_mean_norm=v['predicted_gain_sum']/n,
                actual_gain_mean_norm=v['actual_gain_sum']/n,
                predicted_harm_probability_mean=v['predicted_harm_sum']/n,
                actual_harm_frequency=v['actual_harm_sum']/n,
                gain_mae_norm=v['gain_abs_error_sum']/n))
        return out

class DiagnosticCollector:
    def __init__(self,mean,std,target_idx,mode='behavior',max_cases=60):
        self.mean=float(mean);self.std=float(std);self.target_idx=int(target_idx)
        self.global_stats=Accumulator();self.grouped=GroupedMetrics()
        self.calibration=Calibrator();self.interventions=defaultdict(Accumulator)
        self.utility_stats=defaultdict(float);self.windows=0
        self.cases=[];self.all_windows=[];self.max_cases=max_cases
        self.edge_sum=None;self.edge_seen=None;self.edge_metadata=None
        self.mode=mode
    def update(self,batch,out,idx,window_starts):
        y=arr(batch['y_raw']).astype(np.float64)
        valid=arr(batch['future_valid']).astype(bool)&np.isfinite(y)
        pred={k:arr(v).astype(np.float64)*self.std+self.mean for k,v in fixed_predictions(out).items()}
        # Oracle discrete uses future truth for diagnostic only. Never enters model inputs.
        can=arr(out['candidates']).astype(np.float64)*self.std+self.mean
        oracle=np.min(np.abs(can-y[...,None]),axis=-1)
        pred['oracle_discrete']=np.where(valid,y+oracle,y)
        # oracle has the correct absolute distance but synthetic sign, used only for MAE.
        dm=arr(batch['direct_mask']).astype(bool)
        if dm.ndim==2:dm=np.broadcast_to(dm[:,None,:],valid.shape)
        extras={'support_score':arr(out['support']['support_score'][...,0]),
                'coverage':arr(out['coverage'][...,0]),
                'local_gate':arr(out['policy']['weights'][...,1]+out['policy']['weights'][...,2]),
                'prop_gate':arr(out['policy']['weights'][...,2])}
        trace=out.get('propagation_trace')
        if trace is not None:
            for name in ('null_fraction','weighted_source_age','incoming_available_count','message_l2'):
                extras[name]=arr(trace[name])
        self.global_stats.add(y,pred,valid,**extras)
        self.grouped.add('evaluation_mask','masked',y,pred,valid&dm,**extras)
        self.grouped.add('evaluation_mask','non_masked',y,pred,valid&(~dm),**extras)
        for h in range(valid.shape[1]):
            vm=np.zeros_like(valid);vm[:,h,:]=valid[:,h,:]
            self.grouped.add('horizon',f'H{h+1:02d}',y,pred,vm,**extras)
        extra={'target_idx':self.target_idx}
        if trace is not None:
            extra['incoming_available_count']=arr(trace['incoming_available_count']).mean(1)
            extra['weighted_source_age']=arr(trace['weighted_source_age']).mean(1)
        cats=group_masks(batch,extra)
        for name,values in cats.items():
            for label in np.unique(values):
                gm=(values==label)[:,None,:]&valid
                self.grouped.add(name,str(label),y,pred,gm,**extras)
        weights=arr(out['policy']['weights'])
        ent=-(weights*np.log(weights.clip(1e-12))).sum(-1)
        true_best=np.argmin(np.abs(can-y[...,None]),axis=-1)
        chosen=weights.argmax(-1)
        self.utility_stats['entropy_sum']+=float(ent[valid].sum())
        self.utility_stats['action_correct']+=int(((chosen==true_best)&valid).sum())
        self.utility_stats['n']+=int(valid.sum())
        for a in range(3):
            self.utility_stats[f'p{a}_sum']+=float(weights[...,a][valid].sum())
            self.utility_stats[f'p{a}_argmax']+=int(((chosen==a)&valid).sum())
        y_norm=arr(batch['y_norm'])
        norm_can=arr(out['candidates'])
        err=np.abs(norm_can-y_norm[...,None])
        true_gain=err[...,:1]-err[...,1:]
        true_harm=(err[...,1:]>err[...,:1]+1e-5).astype(float)
        policy=out['policy']
        self.calibration.update(arr(policy['predicted_gain']),arr(policy['predicted_harm_logits']),true_gain,true_harm,valid)
        self.windows+=len(idx)
        for b,i in enumerate(idx):
            mv=valid[b]
            n=int(mv.sum())
            if not n:continue
            base=float(np.abs(pred['base'][b]-y[b])[mv].mean())
            final=float(np.abs(pred['controlled'][b]-y[b])[mv].mean())
            item={'index':int(i),'window_start':int(window_starts[b]),'base_mae':base,
                'controlled_mae':final,'gain_mae':base-final,'valid_positions':n,
                'target_coverage_mean':float((arr(batch['observed_mask'])[b,...,self.target_idx]>0.5).mean())}
            self.cases.append(item)
            self.all_windows.append(item)
        if len(self.cases)>2*self.max_cases:
            self.cases=sorted(self.cases,key=lambda row:abs(row['gain_mae']),reverse=True)[:self.max_cases]
        if trace is not None and 'edge_weights' in trace:
            w=arr(trace['edge_weights'])
            sums=w.sum(axis=(0,1),dtype=np.float64)
            count=w.shape[0]*w.shape[1]
            if self.edge_sum is None:
                self.edge_sum=np.zeros_like(sums,dtype=np.float64)
                self.edge_seen=0
                self.edge_metadata={name:arr(trace[name]) for name in ('src','dst','edge_is_functional','edge_forward','edge_reverse')}
            self.edge_sum+=sums;self.edge_seen+=count
    def update_interventions(self,model,batch,out,idx,names,max_fresh_age=3):
        valid=arr(batch['future_valid']).astype(bool)&np.isfinite(arr(batch['y_raw']))
        y=arr(batch['y_raw']).astype(np.float64)
        for name in names:
            with torch.inference_mode():
                changed,fixed_policy,prop=run_intervention(model,batch,out,name,max_age=max_fresh_age)
            base=arr(out['baseline'])*self.std+self.mean
            original=arr(out['prediction'])*self.std+self.mean
            d={'base':base,'controlled':original}
            # Compare intervention predictions with original on the exact same valid positions.
            for strategy,value in [('recomputed_policy',changed),('fixed_original_policy',fixed_policy),('propagation_only',prop)]:
                predicted=arr(value)*self.std+self.mean
                self.interventions[(name,strategy)].add(y,{'base':original,'controlled':predicted},valid)
    def action_rows(self):
        n=max(self.utility_stats['n'],1)
        return [dict(action=['no_correction','local','local_propagation'][i],
            avg_probability=self.utility_stats[f'p{i}_sum']/n,
            argmax_fraction=self.utility_stats[f'p{i}_argmax']/n,
            argmax_correct_fraction=self.utility_stats['action_correct']/n,
            entropy_mean=self.utility_stats['entropy_sum']/n,
            valid_positions=int(self.utility_stats['n'])) for i in range(3)]
    def intervention_rows(self):
        rows=[]
        for (name,strategy),a in sorted(self.interventions.items()):
            r=a.finish()
            rows.append(dict(intervention=name,strategy=strategy,
                             valid_positions=r.get('valid_positions',0),
                             mae=r.get('controlled_mae'),original_mae=r.get('base_mae'),
                             delta_vs_original=(r['controlled_mae']-r['base_mae']) if 'controlled_mae' in r else None))
        return rows
    def case_rows(self):
        return sorted(self.cases,key=lambda r:abs(r['gain_mae']),reverse=True)[:self.max_cases]
    def edge_rows(self):
        if self.edge_sum is None:return []
        data=self.edge_metadata
        rows=[]
        for i in range(len(self.edge_sum)):
            rows.append({'edge_id':i,'src':int(data['src'][i]),'dst':int(data['dst'][i]),
                         'type':'functional' if data['edge_is_functional'][i] else 'physical',
                         'forward':bool(data['edge_forward'][i]),'reverse':bool(data['edge_reverse'][i]),
                         'mean_weight':self.edge_sum[i]/self.edge_seen})
        return rows
