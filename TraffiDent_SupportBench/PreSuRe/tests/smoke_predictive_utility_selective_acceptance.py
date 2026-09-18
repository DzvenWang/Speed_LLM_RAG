from pathlib import Path
import sys, yaml, torch
R=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(R))
from presure.models import PredictiveUtilityRerankerV09A, SelectedSupportAcceptanceHead
from presure.selective_acceptance import frozen_top1_acceptance_context, selected_target, selective_acceptance_loss, SelectiveAcceptanceAccumulator

cfg=yaml.safe_load((R/'configs/presure.yaml').read_text()); assert cfg['version'] in {'0.9.3','0.9.4'}
B,N,K,P,F=2,5,6,4,23
m=PredictiveUtilityRerankerV09A(F,N,P,encoder_hidden=24,state_dim=8,node_dim=4,sense_dim=6,residual_dim=12,pair_dim=16,hidden=24,horizon_dim=4,attention_heads=4,dropout=0.0)
for p in m.parameters(): p.requires_grad_(False)
q=torch.randn(B,N,F); s=torch.randn(B,N,K,F); qs=torch.randn(B,N,6); ss=torch.randn(B,N,K,6); rr=torch.randn(B,N,K,P); qb=torch.randn(B,P,N); sb=torch.randn(B,N,K,P); cs=torch.randn(B,K)
with torch.no_grad(): role,mu,sigma,blogit=m.forward_features(q,s,qs,ss,rr,qb,sb,cs); r=mu-.5*sigma
cm=torch.ones(B,N,K,dtype=torch.bool); valid=torch.ones(B,N,P,dtype=torch.bool); target=torch.randn(B,N,K,P)*.1
idx,feat,sv=frozen_top1_acceptance_context(role,mu,sigma,blogit,r,cm,valid)
assert idx.shape==(B,N,P) and feat.shape==(B,N,P,108) and sv.shape==(B,N,P)
su=selected_target(target,idx); assert su.shape==(B,N,P)
h=SelectedSupportAcceptanceHead(role_dim=96,context_dim=12,hidden=32,dropout=0.0); logit=h(feat); assert logit.shape==(B,N,P)
loss,parts=selective_acceptance_loss(logit,su,sv,cfg,sv,sv); loss.backward(); assert torch.isfinite(loss)
assert all(p.grad is None for p in m.parameters()), 'Frozen v0.9A ranker received gradients'
acc=SelectiveAcceptanceAccumulator(.02,.02); best=target.max(2).values; accept=sv&(torch.sigmoid(logit.detach())>=.5); acc.update(su.detach(),accept,best,sv); z=acc.summary(); assert 0<=z['coverage']<=1 and 'accepted_positive_utility_rate' in z
# Regression contract: v0.9A.3 is Top1-or-ABSTAIN only; no fallback/veto helper may enter the runner.
runner=(R/'scripts/run_predictive_utility_selective_acceptance_probe.py').read_text()
assert 'sequential_veto_choice' not in runner and 'max_checks' not in runner
assert 'frozen_top1_acceptance_context' in runner and 'selected support utility > 0' in runner
print('SMOKE PredictiveUtilitySelectiveAcceptance v0.9A.3 PASS')
