"""Small streaming research diagnostics, no prediction NPZ is required.

These are descriptive, *hindsight* diagnostics. Oracle and individual utility
use validation/test labels for scoring only; never feed labels to inference.
"""
from __future__ import annotations
import math
import torch

class _Correlation:
    def __init__(self):
        self.n=0;self.sx=self.sy=self.sxx=self.syy=self.sxy=0.
    def update(self,x,y,valid):
        x=x.float()[valid].detach().cpu();y=y.float()[valid].detach().cpu()
        n=x.numel()
        if not n:return
        self.n+=n;self.sx+=float(x.sum());self.sy+=float(y.sum())
        self.sxx+=float((x*x).sum());self.syy+=float((y*y).sum())
        self.sxy+=float((x*y).sum())
    def value(self):
        if self.n<2:return None
        xvar=self.sxx-self.sx*self.sx/self.n
        yvar=self.syy-self.sy*self.sy/self.n
        if xvar<=1e-10 or yvar<=1e-10:return None
        return (self.sxy-self.sx*self.sy/self.n)/math.sqrt(xvar*yvar)

class MechanismDiagnostics:
    def __init__(self):
        self.sums={};self.counts={}
        self.candidate_corr=_Correlation();self.marginal_corr=_Correlation()
        self.risk_curve={t:[0.,0.,0.,0.,0] for t in (0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.)}
        self.auc_pos=torch.zeros(100,dtype=torch.float64)
        self.auc_neg=torch.zeros(100,dtype=torch.float64)
    def add(self,key,values,valid):
        values=values.detach().float(); valid=valid.bool()
        n=int(valid.sum())
        if not n:return
        self.sums[key]=self.sums.get(key,0.)+float(values[valid].sum())
        self.counts[key]=self.counts.get(key,0)+n
    @torch.no_grad()
    def update(self,out,b):
        y=b['y_norm'].float();base=out['base_prediction'].float()
        valid=b['future_valid'].bool()
        k=out['candidate_residuals'].shape[1]
        bound=float(out['residual_bound'])
        avail=out['candidate_available'].bool()&valid[:,None]
        c=out['candidate_residuals'].float()
        individual=bound*torch.tanh(c/bound)
        individual_gain=((base[:,None]-y[:,None]).abs()-
                         (base[:,None]+individual-y[:,None]).abs())
        self.candidate_corr.update(out['candidate_expected_net_gain'],individual_gain,avail)
        self.add('individual_candidate_gain',individual_gain,avail)
        self.add('candidate_positive_utility_rate',(individual_gain>0).float(),avail)
        normal=out['normal_component'].float()
        increment=out['cf_increment'].float()
        normal_proposal=bound*torch.tanh(normal/bound)
        cf_proposal=bound*torch.tanh((normal+increment)/bound)
        cf_gain=((base[:,None]+normal_proposal-y[:,None]).abs()-
                 (base[:,None]+cf_proposal-y[:,None]).abs())
        self.add('cf_increment_gain',cf_gain,avail)
        self.add('cf_positive_increment_rate',(cf_gain>0).float(),avail)
        if 'cf_wrong_increment' in out:
            wrong=bound*torch.tanh((normal+out['cf_wrong_increment'].float())/bound)
            wrong_gain=((base[:,None]+normal_proposal-y[:,None]).abs()-
                        (base[:,None]+wrong-y[:,None]).abs())
            self.add('correct_cf_minus_shuffled_gain',cf_gain-wrong_gain,avail)
            self.add('correct_cf_better_than_shuffled_rate',(cf_gain>wrong_gain).float(),avail)
        initial=out['marginal_reference_proposal'].float()
        loo=out['candidate_loo_proposal'].float()
        gt=((base[:,None]+loo-y[:,None]).abs()-
            (base[:,None]+initial[:,None]-y[:,None]).abs())
        self.marginal_corr.update(out['candidate_marginal_utility_pred'],gt,avail)
        self.add('marginal_gain',gt,avail)
        gain=(base-y).abs()-(out['prediction'].float()-y).abs()
        self.add('final_gain',gain,valid)
        self.add('final_harm_rate',(gain<0).float(),valid)
        self.add('final_gate',out['gate'],valid)
        self.add('correction_coverage',((out['prediction']-base).abs()>1e-4).float(),valid)
        self.add('predicted_harm_brier',(out['harm_probability'].float()-(gain<0).float()).square(),valid)
        # Streaming joint-harm discrimination and harm/coverage tradeoff.
        # Raw labels are used only for diagnostics, never for inference.
        risk=out['harm_probability'].float().clamp(0,1)
        unguarded=(base+out['pre_rejection_correction'].float()-y).abs()
        base_error=(base-y).abs()
        risk_valid=risk[valid].detach().float().cpu()
        harmful=(unguarded[valid]>base_error[valid]).detach().cpu()
        bins=(risk_valid*99).long().clamp(0,99)
        self.auc_pos+=torch.bincount(bins[harmful],minlength=100).double()
        self.auc_neg+=torch.bincount(bins[~harmful],minlength=100).double()
        for threshold,record in self.risk_curve.items():
            active=valid & (risk<=threshold)
            accepted_harm=active & (unguarded>base_error)
            adjusted=torch.where(active,unguarded,base_error)
            record[0]+=int(active.sum())
            record[1]+=int(accepted_harm.sum())
            record[2]+=float(adjusted[valid].sum())
            record[3]+=int(valid.sum())
            record[4]+=float(unguarded[valid].sum())
        self.add('hard_reject_rate',out['reject_correction'].float(),valid)
        # Component-aware diagnosis, using validation/test truths for scoring
        # ONLY. Normal/CF predictions follow the exact nonlinear allocation.
        if 'proposed_normal' in out:
            n=out['proposed_normal'].float();cf=out['proposed_cf'].float()
            n_gain=(base-y).abs()-(base+n-y).abs()
            cf_gain=(base-y).abs()-(base+cf-y).abs()
            both_gain=(base-y).abs()-(base+n+cf-y).abs()
            self.add('normal_component_gain',n_gain,valid)
            self.add('cf_component_gain',cf_gain,valid)
            self.add('component_interaction_gain',both_gain-n_gain-cf_gain,valid)
            self.add('normal_component_harm_rate',(n_gain<0).float(),valid)
            self.add('cf_component_harm_rate',(cf_gain<0).float(),valid)
            for name in ('normal_agreement','cf_agreement','component_conflict',
                         'normal_signal','cf_signal','effective_candidate_count'):
                self.add(name,out[name],valid)
            self.add('normal_intervention_strength',out['normal_gate'],valid)
            self.add('cf_intervention_strength',out['cf_gate'],valid)
            self.add('component_direction_conflict',((n*cf)<0).float(),valid)
            for name,quantity in (('normal',n_gain),('cf',cf_gain)):
                agreement=out['normal_agreement' if name=='normal' else 'cf_agreement']
                for level,mask in (('low',agreement<.33),
                                   ('medium',(agreement>=.33)&(agreement<.67)),
                                   ('high',agreement>=.67)):
                    self.add(f'{name}_gain_{level}_agreement',quantity,valid&mask)
        if 'support_pair_raw' in out:
            support=out['support_pair_raw']
            # Support matching stratification is descriptive, not causal inference.
            overlap=support[...,2][:,:,None].expand_as(individual_gain)
            for name,flag in (('low_overlap',overlap<.2),('medium_overlap',(overlap>=.2)&(overlap<.6)),
                              ('high_overlap',overlap>=.6)):
                self.add('candidate_gain_'+name,individual_gain,avail&flag)

    def risk_coverage_rows(self):
        rows=[]
        for threshold,(accepted,harm,err,total,baseline) in self.risk_curve.items():
            rows.append(dict(harm_probability_cutoff=threshold,
                correction_coverage=accepted/max(total,1),
                accepted_harm_rate=harm/max(accepted,1),
                overall_harm_frequency=harm/max(total,1),
                normalized_MAE_with_rejection=err/max(total,1),
                normalized_MAE_without_rejection=baseline/max(total,1),
                positions=total))
        return rows

    def harm_auc(self):
        pos=self.auc_pos.sum().item();neg=self.auc_neg.sum().item()
        if not pos or not neg:return None
        below=0.;concordant=0.
        for p,n in zip(self.auc_pos,self.auc_neg):
            concordant+=float(p)*(below+float(n)*.5)
            below+=float(n)
        return concordant/(pos*neg)

    def summary(self):
        out={key:(self.sums[key]/self.counts[key] if self.counts[key] else None)
             for key in self.sums}
        out['joint_harm_approx_auc']=self.harm_auc()
        out['candidate_utility_pearson']=self.candidate_corr.value()
        out['marginal_utility_pearson']=self.marginal_corr.value()
        out['n_valid_candidate_positions']=self.counts.get('individual_candidate_gain',0)
        out['n_valid_forecast_positions']=self.counts.get('final_gain',0)
        return out


def select_matched_wrong_mask(mask, family=None, pool_masks=None, pool_family=None):
    """Coverage/family-matched WRONG query masks, with an explicit valid flag.

    Returns (wrong_mask, changed[B]). An identical mask is never silently
    claimed to be a negative control. May return a partial set of valid pairs;
    evaluation records the proportion and excludes unchanged queries from
    its dedicated wrong-mask diagnostic rather than inventing perturbations.
    """
    B=mask.shape[0]
    candidates=pool_masks.to(mask.device) if pool_masks is not None else mask
    flat=mask.bool().reshape(B,-1)
    pool=candidates.bool().reshape(len(candidates),-1)
    if len(candidates)==0:
        return mask.clone(),torch.zeros(B,device=mask.device,dtype=torch.bool)
    coverage_gap=(flat.float().mean(-1)[:,None]-pool.float().mean(-1)[None]).abs()
    different=(flat[:,None]!=pool[None]).any(dim=-1)
    score=coverage_gap
    if family is not None and (pool_family is not None or pool_masks is None):
        family_pool=pool_family if pool_family is not None else family
        labels={key:i for i,key in enumerate(sorted(set(map(str,list(family)+list(family_pool)))))}
        qids=torch.tensor([labels[str(x)] for x in family],device=mask.device)
        pids=torch.tensor([labels[str(x)] for x in family_pool],device=mask.device)
        score=score+(qids[:,None]!=pids[None]).float()*10.
    score=score.masked_fill(~different,float('inf'))
    nearest=score.argmin(-1)
    valid=different.any(dim=-1)
    result=candidates[nearest].clone()
    result[~valid]=mask[~valid]
    return result,valid
