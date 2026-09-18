#!/usr/bin/env python3
from pathlib import Path
import ast,re,yaml
R=Path(__file__).resolve().parents[1]
for p in R.rglob('*.py'): ast.parse(p.read_text(),filename=str(p))
c=yaml.safe_load((R/'configs/presure.yaml').read_text());assert c['version'] in {'0.9.5','0.9.5.1'};assert ('query_error_state_representation_v095' in c['implementation_revision'] or 'query_error_state_representation_diagnostics_v0951' in c['implementation_revision']);assert c['project']['exp_root'].endswith('/PreSuRe_exps');assert int(c['node_routing']['utility_candidate_k'])>=8
for p in R.glob('pbs/*.pbs'):
 s=p.read_text();assert '#PBS -j oe' in s and '/PreSuRe_logs/out/' in s and 'presure_live_log_start' in s
 if '#PBS -q gpuvolta' in s:
  mg=re.search(r'^#PBS -l ngpus=(\d+)\s*$',s,re.M);mc=re.search(r'^#PBS -l ncpus=(\d+)\s*$',s,re.M);assert mg and mc;assert int(mc.group(1))==12*int(mg.group(1))
rv=(R/'scripts/run_variant.py').read_text()
for name in ['NodeUsefulnessNet','m3_train_diagnostics','m3_select_for_pairwise_router','m4_soft_teacher_targets','m4_soft_distill_loss','m4_pairwise_utility_rank_loss','continuous_gate_teacher','continuous_gate_loss','router_use_threshold']: assert name in rv
assert 'calibrate_node_alpha' not in rv and 'alpha_n' not in rv
models=(R/'presure/models.py').read_text()
for x in ['PairwiseSupportRouter','MultiheadAttention','self.qctx','self.sctx','self.residual','support_base_norm']: assert x in models
assert 'direct_mask' not in models and 'support_mask' not in models
pipe=(R/'presure/pipeline.py').read_text();assert 'pairwise_support_router_v07' in pipe
ld=(R/'scripts/run_local_diagnostics.py').read_text()
for x in ['E24','E_M3TOP4','D_M3TOP4','SOFTMIX_M3TOP4','np.array(a,copy=True)']: assert x in ld
full=(R/'pbs/full_pipeline.pbs').read_text();assert 'SOFTMIX_M3TOP4' in full and 'E_M3TOP4' in full and 'D_M3TOP4' in full
submit_dev=(R/'scripts/submit_development.sh').read_text();submit_formal=(R/'scripts/submit_formal.sh').read_text();assert submit_dev.count('qsub')==1 and submit_formal.count('qsub')==1
mg=re.search(r'^#PBS -l ngpus=(\d+)\s*$',full,re.M);mc=re.search(r'^#PBS -l ncpus=(\d+)\s*$',full,re.M);assert mg and int(mg.group(1))==1 and mc and int(mc.group(1))==12
probe=(R/'scripts/run_predictive_need_probe.py').read_text(); assert 'PredictiveNeedNet' in probe and "for mode in ['A','B','C']" in probe and 'GO_V08' in probe
pbs=(R/'pbs/predictive_need_probe.pbs').read_text(); assert 'run_predictive_need_probe.py' in pbs and '#PBS -l ngpus=1' in pbs
state=(R/'scripts/run_predictive_need_state_probe.py').read_text(); assert 'PredictiveNeedStateNet' in state and 'need_q75' in state and 'TRUE_STATE_ANALYTIC' in state and 'GO_V08_STATE' in state
spbs=(R/'pbs/predictive_need_state_probe.pbs').read_text(); assert 'LOGROOT=' in spbs and 'run_predictive_need_state_probe.py' in spbs and '#PBS -l ngpus=1' in spbs
gate=(R/'scripts/run_need_gated_utility_probe.py').read_text(); assert 'NeedGatedUtilityProbe' in gate and 'top10' in gate and 'train_only_policy_calibration' in gate
gpbs=(R/'pbs/need_gated_utility_probe.pbs').read_text(); assert 'LOGROOT=' in gpbs and 'run_need_gated_utility_probe.py' in gpbs and '#PBS -l ngpus=1' in gpbs
assert c['need_gated_utility_probe']['reuse_only'] is True
spec=(R/'scripts/run_need_conditioned_utility_specialist_probe.py').read_text(); assert 'NeedConditionedUtilitySpecialistProbe' in spec and 'train_specialist' in spec and 'two_expert' in spec
sp=(R/'pbs/need_conditioned_utility_specialist_probe.pbs').read_text(); assert 'LOGROOT=' in sp and 'run_need_conditioned_utility_specialist_probe.py' in sp and '#PBS -l ngpus=1' in sp
assert c['need_conditioned_utility_specialist_probe']['initialize_from_utility_a'] is True
route=(R/'scripts/run_deployment_aligned_utility_router_probe.py').read_text(); assert 'DeploymentAlignedUtilityRouterProbe' in route and 'train_deployment_specialist' in route and 'ExpertPreferenceRouter' in route
rpbs=(R/'pbs/deployment_aligned_utility_router_probe.pbs').read_text(); assert 'LOGROOT=' in rpbs and 'run_deployment_aligned_utility_router_probe.py' in rpbs and '#PBS -l ngpus=1' in rpbs
assert c['deployment_aligned_utility_router_probe']['reuse_v074_hard_specialist'] is True
meta=(R/'scripts/run_regret_aware_support_proposal_meta_probe.py').read_text(); assert 'RegretAwareSupportProposalMetaRankingProbe' in meta and 'SupportProposalMetaRanker' in meta and 'oracle_proposal' in meta
mpbs=(R/'pbs/regret_aware_support_proposal_meta_probe.pbs').read_text(); assert 'LOGROOT=' in mpbs and 'run_regret_aware_support_proposal_meta_probe.py' in mpbs and '#PBS -l ngpus=1' in mpbs
assert c['proposal_meta_ranking_probe']['reuse_v075_experts'] is True
sel=(R/'scripts/run_predictive_utility_selective_acceptance_probe.py').read_text(); assert 'Selective Utility Acceptance' in sel and 'fallback_used' in sel and 'top1_only' in sel
sp=(R/'pbs/predictive_utility_selective_acceptance_probe.pbs').read_text(); assert 'PreUtilA3' in sp and 'no fallback / no re-ranking' in sp
print('PRESURE V0.9.5 QUERY ERROR-STATE STATIC CONTRACT SMOKE PASS')
