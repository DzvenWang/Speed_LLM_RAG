"""Read-only, streamed paired mechanism diagnostics for frozen TCRR v5.3.

Strictly descriptive: uses labels AFTER predictions; oracle and selected-candidate
scores must never influence checkpoint selection, inference or calibration.
All forecast errors are in the original target units. Only small per-query
summaries and aggregated tables are retained (no B*K*P*N disk export).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


class _Aggregate:
    def __init__(self):
        self.n = 0
        self.gain_sum = 0.0
        self.harm_count = 0
        self.harm_loss_sum = 0.0
        self.benefit_sum = 0.0
        self.base_error_sum = 0.0
        self.c12_error_sum = 0.0
        self.c3_error_sum = 0.0

    def add(self, gain, base_error, c12_error, c3_error, mask):
        m = mask.bool()
        n = int(m.sum())
        if not n:
            return
        g = gain[m].detach().float()
        self.n += n
        self.gain_sum += float(g.sum())
        self.harm_count += int((g < -1e-6).sum())
        self.harm_loss_sum += float((-g).clamp_min(0).sum())
        self.benefit_sum += float(g.clamp_min(0).sum())
        self.base_error_sum += float(base_error[m].sum())
        self.c12_error_sum += float(c12_error[m].sum())
        self.c3_error_sum += float(c3_error[m].sum())

    def row(self):
        n = max(self.n, 1)
        return dict(valid_positions=self.n, c3_mae_gain=self.gain_sum/n,
                    c3_harm_rate=self.harm_count/n,
                    c3_harmful_loss_per_position=self.harm_loss_sum/n,
                    c3_benefit_per_position=self.benefit_sum/n,
                    baseline_mae=self.base_error_sum/n,
                    c12_mae=self.c12_error_sum/n, c3_mae=self.c3_error_sum/n)


class _BinnedAUC:
    """Approximate AUROC and calibration for predicted risk, without storing points."""
    def __init__(self, bins=100):
        self.bins = bins
        self.pos = torch.zeros(bins, dtype=torch.float64)
        self.neg = torch.zeros(bins, dtype=torch.float64)
        self.risk_sum = torch.zeros(bins, dtype=torch.float64)
        self.count = torch.zeros(bins, dtype=torch.float64)
        self.brier_sum = 0.0

    def update(self, risk, label, valid):
        r = risk.detach().float()[valid].clamp(0,1).cpu()
        y = label.detach().bool()[valid].cpu()
        if not r.numel():
            return
        bin_id = (r * (self.bins - 1)).long()
        self.pos += torch.bincount(bin_id[y], minlength=self.bins).double()
        self.neg += torch.bincount(bin_id[~y], minlength=self.bins).double()
        self.count += torch.bincount(bin_id, minlength=self.bins).double()
        self.risk_sum += torch.bincount(bin_id, weights=r.double(), minlength=self.bins)
        self.brier_sum += float(((r - y.float()) ** 2).sum())

    def summary(self):
        p = float(self.pos.sum()); n = float(self.neg.sum())
        auc = None
        if p > 0 and n > 0:
            below = 0.; concordant = 0.
            for pp, nn in zip(self.pos, self.neg):
                concordant += float(pp) * (below + float(nn) / 2)
                below += float(nn)
            auc = concordant / (p * n)
        total = p+n
        ece = None
        if total:
            ece = float((self.count/self.count.sum() *
                  ((self.risk_sum/self.count.clamp_min(1)) -
                   (self.pos/self.count.clamp_min(1))).abs()).sum())
        return dict(auroc_approx_100bins=auc, brier=self.brier_sum/total if total else None,
                    ece_approx_100bins=ece, positive_rate=p/total if total else None,
                    positions=int(total))


def _safe_div(a, b):
    return float(a/b) if b else None


def _query_avg(field, valid):
    """One value per query, NaN for queries without valid targets."""
    return ((torch.where(valid,field,torch.zeros_like(field))).sum(dim=(1,2)) /
            valid.sum(dim=(1,2)).clamp_min(1)).detach().float().cpu().tolist()


class P0DeepDiagnostics:
    """Accumulation unit: same query, same controller checkpoint and four modes."""
    REQUIRED = ('M000', 'M100', 'M110', 'M111')

    def __init__(self, mean, std):
        if std <= 0:
            raise ValueError('Invalid target standard deviation')
        self.mean = float(mean)
        self.std = float(std)
        self.query_rows = []
        self.groups = defaultdict(_Aggregate)
        self.risk_base = _BinnedAUC()
        self.risk_c3 = _BinnedAUC()
        self.global_totals = defaultdict(float)
        self.global_counts = defaultdict(int)
        self.candidate_corr = [0., 0., 0., 0., 0., 0.]
        self.candidate_pairwise_correct = 0
        self.candidate_pairwise_total = 0
        self.seen_queries = set()
        self.risk_cutoffs = tuple(i / 20 for i in range(21))
        self.risk_curve = {cut:defaultdict(float) for cut in self.risk_cutoffs}
        self.valid_positions = 0
        self.candidate_positions = 0

    def _mean(self, name, value, mask):
        n = int(mask.sum())
        if not n:
            return
        self.global_totals[name] += float(value.detach().float()[mask].sum())
        self.global_counts[name] += n

    def _group(self, name, bucket, delta, err0, err110, err111, mask):
        self.groups[(name, str(bucket))].add(delta,err0,err110,err111,mask)

    @torch.no_grad()
    def update(self, outputs, batch, families=None, retrieval_scores=None):
        if not all(m in outputs for m in self.REQUIRED):
            raise ValueError('P0 requires paired M000 M100 M110 M111 outputs')
        m = outputs
        base = m['M110']['base_prediction'].float()
        y = batch['y_raw'].float()
        valid = batch['future_valid'].bool()
        shape = tuple(base.shape)
        if valid.shape != base.shape or y.shape != base.shape or len(shape) != 3:
            raise ValueError('Expected matching [B,P,N] base, y_raw and future_valid')
        if not torch.isfinite(y[valid]).all():
            raise ValueError('Nonfinite target in valid evaluation positions')
        ids = batch['index'].detach().cpu().tolist()
        if len(ids) != len(set(ids)) or any(int(i) in self.seen_queries for i in ids):
            raise ValueError('P0 requires each dataset query index exactly once')
        self.seen_queries.update(map(int,ids))
        b, p, n = shape
        if families is None:
            families = ['unknown']*b
        if len(families) != b:
            raise ValueError('Fault family/query alignment mismatch')
        def raw(t):return t.float()*self.std+self.mean
        def err(t):return (raw(t)-y).abs()
        error_base = err(base)
        e000 = err(m['M000']['prediction'])
        e100 = err(m['M100']['prediction'])
        e110 = err(m['M110']['prediction'])
        e111 = err(m['M111']['prediction'])
        g1 = e000-e100
        g2 = e100-e110
        g3 = e110-e111
        gain_base = error_base-e111
        self.valid_positions += int(valid.sum())
        self._mean('c1_gain_M000_to_M100',g1,valid)
        self._mean('c2_gain_M100_to_M110',g2,valid)
        self._mean('c3_gain_M110_to_M111',g3,valid)
        self._mean('m111_gain_over_backbone',gain_base,valid)
        self._mean('c3_harm_rate',(g3 < -1e-6).float(),valid)
        self._mean('c3_harm_magnitude',(-g3).clamp_min(0),valid)
        self._mean('c3_benefit_magnitude',g3.clamp_min(0),valid)
        self._mean('c3_harms_previously_beneficial_M110',((g3 < -1e-6)&(e110 < error_base-1e-6)).float(),valid)
        self._mean('c3_harms_previously_harmful_M110',((g3 < -1e-6)&(e110 > error_base+1e-6)).float(),valid)
        o = m['M111']
        self._mean('c3_normal_gate',o['normal_gate'].float(),valid)
        self._mean('c3_cf_gate',o['cf_gate'].float(),valid)
        self._mean('c3_hard_reject_rate',o['reject_correction'].float(),valid)
        self._mean('c3_component_conflict',o['component_conflict'].float(),valid)
        risk = o['harm_probability'].float().clamp(0,1)
        # Risk Head predicts harm relative to Backbone, not the causal effect of C3.
        pre = base+o['pre_rejection_correction'].float()
        pre_error = err(pre)
        self.risk_base.update(risk,pre_error > error_base+1e-6,valid)
        self.risk_c3.update(risk,g3 < -1e-6,valid)
        n_valid=int(valid.sum())
        for threshold in self.risk_cutoffs:
            accept=valid & (risk<=threshold)
            row=self.risk_curve[threshold]
            row['total']+=n_valid
            row['accepted']+=int(accept.sum())
            row['accepted_base_harm']+=int((accept&(pre_error>error_base+1e-6)).sum())
            row['accepted_c3_harm']+=int((accept&(g3 < -1e-6)).sum())
            row['backbone_error_sum']+=float(error_base[valid].sum())
            row['m110_error_sum']+=float(e110[valid].sum())
            row['pre_c3_error_sum']+=float(pre_error[valid].sum())
            # Counterfactual threshold policies, not deployed decisions.
            row['hybrid_vs_backbone_error_sum']+=float(torch.where(accept,pre_error,error_base)[valid].sum())
            row['hybrid_vs_m110_error_sum']+=float(torch.where(accept,e111,e110)[valid].sum())
        obs = batch['mask'].float().reshape(b,-1).mean(1)
        obs_edges = (0.,.1,.25,.5,.75,1.000001)
        for h in range(p):
            mm = torch.zeros_like(valid);mm[:,h,:] = valid[:,h,:]
            self._group('horizon',h+1,g3,error_base,e110,e111,mm)
        for j in range(len(obs_edges)-1):
            selector = (obs>=obs_edges[j]) & (obs<obs_edges[j+1])
            self._group('observation_ratio',f'{obs_edges[j]:.2f}-{obs_edges[j+1]:.2f}',
                        g3,error_base,e110,e111,valid & selector[:,None,None])
        conflict = o['component_conflict'].float()>.5
        self._group('component_conflict','conflict',g3,error_base,e110,e111,valid&conflict)
        self._group('component_conflict','aligned_or_neutral',g3,error_base,e110,e111,valid&~conflict)
        if 'direct_mask' in batch:
            direct=batch['direct_mask'].bool()
            if direct.shape != valid.shape:
                raise ValueError('Expected direct_mask [B,P,N]')
            self._group('direct_mask','direct',g3,error_base,e110,e111,valid&direct)
            self._group('direct_mask','not_direct',g3,error_base,e110,e111,valid&~direct)
        norm_magnitude = m['M110']['proposed_correction'].float().abs()
        for j,(low,high) in enumerate(((0,.1),(.1,.5),(.5,1.),(1.,3.),(3.,float('inf')))):
            sel = (norm_magnitude>=low)&(norm_magnitude<high)
            self._group('proposal_magnitude_norm',f'{low:g}-{high:g}',g3,error_base,e110,e111,valid&sel)
        for i,f in enumerate(families):
            mask_i=torch.zeros_like(valid);mask_i[i]=valid[i]
            self._group('fault_family',f,g3,error_base,e110,e111,mask_i)

        # C2 candidate data are from M110; each single candidate is bounded
        # with the same tanh bound as C2 and None is always available.
        c2=m['M110']; c=c2['candidate_residuals'].float()
        avail=c2['candidate_available'].bool() & valid[:,None]
        if c.ndim!=4 or c.shape[0]!=b or c.shape[2:]!=shape[1:]:
            raise ValueError('Expected candidate residuals [B,K,P,N]')
        k=c.shape[1]
        w=c2['candidate_weights'].float()
        if w.shape != (b,k+1,p,n):
            raise ValueError('Expected candidate_weights [B,K+1,P,N]')
        bound=float(c2['residual_bound'])
        if bound <= 0:
            raise ValueError('Invalid correction bound')
        individual = bound*torch.tanh(c/bound)
        individual_error = (raw(base[:,None]+individual)-y[:,None]).abs()
        cand_gain = error_base[:,None]-individual_error
        cand_n = int(avail.sum())
        self.candidate_positions += cand_n
        self._mean('candidate_positive_rate',(cand_gain>1e-6).float(),avail)
        true_gain = cand_gain[avail]
        predicted = c2['candidate_expected_net_gain'].float()[avail]
        if true_gain.numel():
            # Accumulate correlation in source units, no gigantic CSV.
            pred_cpu=predicted.double(); true_cpu=true_gain.double()
            moments=[float(pred_cpu.sum()),float(true_cpu.sum()),
                     float(pred_cpu.square().sum()),float(true_cpu.square().sum()),
                     float((pred_cpu*true_cpu).sum()),float(pred_cpu.numel())]
            self.candidate_corr=[x+y for x,y in zip(self.candidate_corr,moments)]
        # Pairwise within-query/horizon/node candidate ranking concordance.
        # Only legal pairs with unequal true/predicted utility count.
        if k > 1:
            idx_i,idx_j=torch.triu_indices(k,k,offset=1,device=c.device)
            pair_valid=avail[:,idx_i]&avail[:,idx_j]
            pdiff=c2['candidate_expected_net_gain'].float()[:,idx_i]-c2['candidate_expected_net_gain'].float()[:,idx_j]
            tdiff=cand_gain[:,idx_i]-cand_gain[:,idx_j]
            comparable=pair_valid & (pdiff.abs()>1e-6) & (tdiff.abs()>1e-6)
            self.candidate_pairwise_total+=int(comparable.sum())
            self.candidate_pairwise_correct+=int(((pdiff*tdiff>0)&comparable).sum())
        best_err = torch.minimum(error_base, individual_error.masked_fill(~avail,float('inf')).min(1).values)
        oracle_gain = error_base-best_err
        selected = w.argmax(dim=1)
        selected_none=(selected==k)
        chosen_err = individual_error.gather(1,selected.clamp_max(k-1).unsqueeze(1)).squeeze(1)
        chosen_err = torch.where(selected_none,error_base,chosen_err)
        top1_gain = error_base-chosen_err
        pred_score = c2['candidate_expected_net_gain'].float().masked_fill(~avail,-float('inf'))
        top_utility=pred_score.argmax(dim=1)
        utility_err=individual_error.gather(1,top_utility.unsqueeze(1)).squeeze(1)
        utility_err=torch.where(avail.any(1),utility_err,error_base)
        top_utility_gain=error_base-utility_err
        score_gain=None
        if retrieval_scores is not None:
            if retrieval_scores.shape != (b,k):
                raise ValueError('Expected retrieval scores [B,K]')
            # Transparent, *no-trained-utility* top-1 heuristic, using the same
            # already-retrieved shortlist. Unlike C2 it does not learn a None
            # decision: not a like-for-like deployment comparator.
            score_map=retrieval_scores.float()[:,:,None,None].expand_as(c)
            score_idx=score_map.masked_fill(~avail,-float('inf')).argmax(1)
            score_err=individual_error.gather(1,score_idx[:,None]).squeeze(1)
            score_err=torch.where(avail.any(1),score_err,error_base)
            score_gain=error_base-score_err
        none_weight=w[:,-1]
        harmful_w=(w[:,:k]*((cand_gain < -1e-6)&avail).float()).sum(1)
        active_w=w[:,:k].sum(1)
        harmful_weight_share=harmful_w/active_w.clamp_min(1e-8)
        initial = c2['initial_proposal'].float()
        initial_gain = error_base-err(base+initial)
        self._mean('candidate_oracle_single_gain_over_backbone',oracle_gain,valid)
        self._mean('candidate_top_weight_gain_over_backbone',top1_gain,valid)
        self._mean('candidate_top_predicted_utility_gain',top_utility_gain,valid)
        self._mean('candidate_top_predicted_utility_positive_rate',(top_utility_gain>1e-6).float(),valid)
        if score_gain is not None:
            self._mean('candidate_top_retrieval_score_gain',score_gain,valid)
            self._mean('candidate_top_retrieval_score_positive_rate',(score_gain>1e-6).float(),valid)
        self._mean('c2_final_mixture_gain_over_backbone',error_base-e110,valid)
        self._mean('candidate_initial_mixture_gain_over_backbone',initial_gain,valid)
        # Signed gap, not regret: a mixture may outperform all single candidates.
        self._mean('oracle_single_minus_final_mixture',e110-best_err,valid)
        self._mean('oracle_single_minus_top_weight',chosen_err-best_err,valid)
        self._mean('top_weight_matches_single_oracle',((chosen_err-best_err).abs()<1e-5).float(),valid)
        self._mean('support_none_weight',none_weight,valid)
        self._mean('support_harmful_weight_share',harmful_weight_share,valid & (active_w>1e-8))
        self._mean('top_weight_candidate_positive_rate',(top1_gain>1e-6).float(),valid)
        self._mean('oracle_single_positive_rate',(oracle_gain>1e-6).float(),valid)
        qmetrics=dict(c1_gain=_query_avg(g1,valid),c2_gain=_query_avg(g2,valid),
                      c3_gain=_query_avg(g3,valid),
                      c3_harm_rate=_query_avg((g3 < -1e-6).float(),valid),
                      m110_benefit_over_base=_query_avg(error_base-e110,valid),
                      oracle_single_gain=_query_avg(oracle_gain,valid),
                      oracle_single_minus_m110=_query_avg(e110-best_err,valid),
                      top_weight_gain=_query_avg(top1_gain,valid),
                      top_retrieval_score_gain=_query_avg(score_gain if score_gain is not None else torch.zeros_like(base),valid),
                      none_weight=_query_avg(none_weight,valid),
                      harmful_support_weight_share=_query_avg(harmful_weight_share,valid),
                      harm_probability=_query_avg(risk,valid),
                      normal_gate=_query_avg(o['normal_gate'].float(),valid),
                      cf_gate=_query_avg(o['cf_gate'].float(),valid),
                      m110_mae=_query_avg(e110,valid),m111_mae=_query_avg(e111,valid))
        obs_list=obs.detach().cpu().tolist()
        counts=valid.sum((1,2)).cpu().tolist()
        for i,q in enumerate(ids):
            record=dict(dataset_index=int(q),fault_family=str(families[i]),
                        observed_ratio=float(obs_list[i]),valid_positions=int(counts[i]))
            record.update({key:float(values[i]) for key,values in qmetrics.items()})
            self.query_rows.append(record)

    def summary(self):
        n=len(self.query_rows)
        if not n or not self.valid_positions:
            raise RuntimeError('No valid P0 diagnostic positions')
        a,b,aa,bb,ab,count=self.candidate_corr
        cov=ab-a*b/count if count else 0.
        vx=aa-a*a/count if count else 0.
        vy=bb-b*b/count if count else 0.
        corr=cov/(vx*vy)**.5 if count>=2 and vx>0 and vy>0 else None
        return dict(queries=n,valid_positions=self.valid_positions,
                    candidate_valid_positions=self.candidate_positions,
                    metrics={k:_safe_div(v,self.global_counts[k]) for k,v in sorted(self.global_totals.items())},
                    candidate_expected_gain_pearson=corr,
                    candidate_pairwise_ranking_accuracy=_safe_div(self.candidate_pairwise_correct,self.candidate_pairwise_total),
                    candidate_pairwise_valid_comparisons=self.candidate_pairwise_total,
                    harm_risk_vs_backbone=self.risk_base.summary(),
                    harm_risk_vs_c3_off=self.risk_c3.summary(),
                    important_caveats=[
                        'Oracle single-candidate gains use future targets only for hindsight; never usable as a deployable policy.',
                        'C3 harm is paired M110 vs M111 on the same controller checkpoint. Risk head predicts harm vs Backbone, not C3 marginal harm.',
                        'Signed oracle-single-minus-mixture gap is not nonnegative regret: a mixture may beat all single candidates.',
                        'Full test labels are diagnostic-only; any model or threshold change must be selected without test labels.',
                        'Position-level observations are correlated in time and space; use query/block-level uncertainty intervals.',
                        'C1/C2 toggle effects on controller-stage weights are shared-weight ablations, not retrained models.',
                        'Retrieval top-score is a simple top-one heuristic without a trained None decision, not a full deployment comparator.',
                        'Risk-coverage fallback to M110 is a hypothetical diagnostic comparator, not a supported deployed C3 rejection action.',
                    ])

    def save(self,folder,metadata):
        folder=Path(folder)
        folder.mkdir(exist_ok=True,parents=True)
        def csv_write(path, rows):
            if not rows:return
            with path.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
        csv_write(folder/'per_query.csv',self.query_rows)
        grouped=[]
        for (kind,key), agg in sorted(self.groups.items()):
            grouped.append(dict(group_kind=kind,group=key,**agg.row()))
        csv_write(folder/'grouped_c3.csv',grouped)
        curve=[]
        for cutoff,row in self.risk_curve.items():
            total=max(row['total'],1); accepted=max(row['accepted'],1)
            curve.append(dict(harm_cutoff=cutoff,coverage=row['accepted']/total,
                accepted_base_harm_rate=row['accepted_base_harm']/accepted,
                accepted_c3_harm_rate=row['accepted_c3_harm']/accepted,
                backbone_mae=row['backbone_error_sum']/total,
                m110_mae=row['m110_error_sum']/total,
                unthresholded_pre_c3_mae=row['pre_c3_error_sum']/total,
                hypothetical_mae_fallback_backbone=row['hybrid_vs_backbone_error_sum']/total,
                hypothetical_mae_fallback_m110=row['hybrid_vs_m110_error_sum']/total,
                positions=int(row['total'])))
        csv_write(folder/'hypothetical_risk_coverage.csv',curve)
        examples=sorted(self.query_rows,key=lambda r:r['c3_gain'])[:30]
        csv_write(folder/'worst_c3_queries.csv',examples)
        report=dict(schema='TCRR_v5.3.0_P0_v1',metadata=metadata,**self.summary())
        (folder/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        return report
