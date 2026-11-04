"""C1 candidate utility / Normal-CF headroom, strictly Selection-only labels.

The candidate oracle compares the actually transported, bounded candidate
corrections on the SAME query future-valid set. Invalid candidate elements
receive zero correction and thus contribute baseline error, not fake zeros.
"""
from __future__ import annotations
import torch
from .metrics import MaskedMetric,SampleDistribution

class CandidateAudit:
    def __init__(self,min_coverage=.50):
        self.min_coverage=float(min_coverage)
        self.metric={k:MaskedMetric() for k in ('teacher_backbone','candidate_oracle_query','candidate_oracle_element',
            'utility_top1','transport_normal_only','transport_normal_plus_cf')}
        self.n_queries=0;self.eligible_queries=0;self.oracle_beneficial=0;self.utility_top1_beneficial=0
        self.utility_top1_harmful=0;self.hit_best=0;self.false_none=0;self.high_none=0
        self.regret_sum=0.;self.gain_sum=0.;self.gain_num=0
        self.rank_agreement=0.;self.rank_count=0
        self.none=SampleDistribution(seed=41);self.estimated_gain=SampleDistribution(seed=42)
        self.true_gain=SampleDistribution(seed=43);self.cf_gain_sum=0.;self.cf_gain_den=0
    def add(self,b,out,transport,mean,std):
        base=out['base_prediction'].float();target=b['y_norm'].float();valid=b['future_valid'].bool()
        avail=out['candidate_available'].bool() & valid[:,None]
        raw=out['candidate_residuals'].float()
        bound=float(out['residual_bound'])
        cand_corr=bound*torch.tanh(raw/bound)
        candidate=base[:,None]+cand_corr
        baseline_abs=(base-target).abs();err=(candidate-target[:,None]).abs()
        count=valid.sum((1,2)).clamp_min(1)
        cand_gain=((baseline_abs[:,None]-err)*valid[:,None]).sum((2,3))/count[:,None]
        coverage=avail.sum((2,3))/count[:,None]
        permitted=coverage>=self.min_coverage
        permitted&=avail.any((2,3))
        # If none eligible, oracle falls back to the original backbone.
        best_gain,best_idx=cand_gain.masked_fill(~permitted,-float('inf')).max(1)
        any_eligible=permitted.any(1)
        best_gain=torch.where(any_eligible,best_gain,torch.zeros_like(best_gain))
        best_gain=best_gain.clamp_min(0)
        query_pred=base+torch.gather(cand_corr,1,best_idx[:,None,None,None].expand(-1,1,base.shape[1],base.shape[2])).squeeze(1)
        query_pred=torch.where((best_gain>0)[:,None,None],query_pred,base)
        elem_corr=torch.where(avail,cand_corr,torch.zeros_like(cand_corr))
        elem_abs=(base[:,None]+elem_corr-target[:,None]).abs()
        good_elem=elem_abs.argmin(1)
        best_elem_abs=torch.minimum(elem_abs.min(1).values,baseline_abs)
        # Build an oracle prediction with correct sign, selected only for offline metric.
        elem_preds=base[:,None]+elem_corr
        selected_elem=torch.gather(elem_preds,1,good_elem[:,None]).squeeze(1)
        selected_elem=torch.where((elem_abs.min(1).values<baseline_abs),selected_elem,base)
        est=out['candidate_transfer_gain'].float()-out['candidate_transfer_harm'].float()
        est_query=(est*avail).sum((2,3))/avail.sum((2,3)).clamp_min(1)
        est_query=est_query.masked_fill(~permitted,-float('inf'))
        pred_idx=est_query.argmax(1)
        pred_corr=torch.gather(cand_corr,1,pred_idx[:,None,None,None].expand(-1,1,base.shape[1],base.shape[2])).squeeze(1)
        top1=base+torch.where(any_eligible[:,None,None],pred_corr,torch.zeros_like(pred_corr))
        top1_gain=cand_gain.gather(1,pred_idx[:,None]).squeeze(1)
        self.metric['teacher_backbone'].add(base*std+mean,target*std+mean,valid)
        self.metric['candidate_oracle_query'].add(query_pred*std+mean,target*std+mean,valid)
        self.metric['candidate_oracle_element'].add(selected_elem*std+mean,target*std+mean,valid)
        self.metric['utility_top1'].add(top1*std+mean,target*std+mean,valid)
        tr_normal=base+bound*torch.tanh(transport['raw_normal'].float()/bound)
        tr_both=base+transport['proposed_correction'].float()
        self.metric['transport_normal_only'].add(tr_normal*std+mean,target*std+mean,valid)
        self.metric['transport_normal_plus_cf'].add(tr_both*std+mean,target*std+mean,valid)
        B=len(base);self.n_queries+=B;self.eligible_queries+=int(any_eligible.sum())
        self.oracle_beneficial+=int((best_gain>0).sum());self.utility_top1_beneficial+=int((top1_gain>0).logical_and(any_eligible).sum())
        self.utility_top1_harmful+=int((top1_gain<0).logical_and(any_eligible).sum())
        self.hit_best+=int(((pred_idx==best_idx)&any_eligible).sum())
        self.regret_sum+=float(((best_gain-top1_gain)*any_eligible).sum());self.gain_sum+=float(best_gain.sum());self.gain_num+=B
        none=out['none_weight'].float();no=none.mean((1,2));self.none.add(no)
        self.high_none+=int((no>.5).sum());self.false_none+=int(((no>.5)&(best_gain>0)).sum())
        actual=cand_gain[permitted].detach();prediction=est_query[permitted].detach()
        if actual.numel():
            self.true_gain.add(actual);self.estimated_gain.add(prediction)
        # Pairwise ranking agreement across legal candidate pairs; query averaged.
        for j in range(B):
            m=permitted[j];v=cand_gain[j,m];u=est_query[j,m]
            if len(v)>1:
                tri=torch.triu_indices(len(v),len(v),1,device=v.device)
                dx=v[tri[0]]-v[tri[1]];dy=u[tri[0]]-u[tri[1]]
                active=(dx.abs()>1e-6)&(dy.abs()>1e-6)
                if active.any():self.rank_agreement+=float(((dx[active]*dy[active])>0).sum());self.rank_count+=int(active.sum())
        cf_benefit=((tr_normal-target).abs()-(tr_both-target).abs())*valid
        self.cf_gain_sum+=float(cf_benefit.sum())*std;self.cf_gain_den+=int(valid.sum())
    def to_dict(self):
        return {'candidate_min_coverage':self.min_coverage,'queries':self.n_queries,
            'eligible_query_fraction':self.eligible_queries/max(1,self.n_queries),
            'oracle_beneficial_query_fraction':self.oracle_beneficial/max(1,self.n_queries),
            'utility_top1_beneficial_fraction':self.utility_top1_beneficial/max(1,self.eligible_queries),
            'utility_top1_harmful_fraction':self.utility_top1_harmful/max(1,self.eligible_queries),
            'utility_top1_hit_best_fraction':self.hit_best/max(1,self.eligible_queries),
            'selection_regret_mean_norm_units':self.regret_sum/max(1,self.eligible_queries),
            'best_candidate_gain_mean_norm_units':self.gain_sum/max(1,self.gain_num),
            'high_none_query_fraction':self.high_none/max(1,self.n_queries),
            'false_none_query_fraction':self.false_none/max(1,self.n_queries),
            'candidate_pairwise_ranking_agreement':self.rank_agreement/max(1,self.rank_count),
            'cf_marginal_gain_raw_per_valid_element':self.cf_gain_sum/max(1,self.cf_gain_den),
            'none_weight':self.none.to_dict(), 'predicted_gain':self.estimated_gain.to_dict(),
            'observed_candidate_gain':self.true_gain.to_dict(),
            'mae_by_mode':{k:v.to_dict() for k,v in self.metric.items()},
            'note':'Best candidate and oracle use Selection future labels solely for offline diagnosis; never available to inference. Candidate scores use unmodified trained C1 transport.'}
