#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,pandas as pd
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe');ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml
ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development');a=ap.parse_args();cfg=load_yaml(ROOT/'configs/presure.yaml');rows=[]
variants=['OracleA','OracleB','OracleC','OracleE','OracleF','OracleD','DiagOracleE24','M0','M1','M2','M3','DiagOracleE_M3Top4','DiagOracleD_M3Top4','DiagOracleSoftMix_M3Top4','M4','M5']
for v in variants:
    r=Path(cfg['project']['exp_root'])/v/cfg['experiment']['subgraph']/a.stage;fs=sorted(r.glob('*/summary.json'),key=lambda p:p.stat().st_mtime) if r.exists() else []
    if not fs: continue
    j=json.loads(fs[-1].read_text())
    if 'oracle_unfiltered_top4' in j:
        m=j['oracle_unfiltered_top4'];ri=j.get('relative_improvement_unfiltered',{});note='unfiltered_top4'
    else:
        m=j.get('validation') or j.get('oracle') or {};ri=j.get('relative_improvement_over_M0') or j.get('relative_improvement') or {};note=''
    diag=j.get('validation_diagnostics_future_labels_only') or j.get('train_only_diagnostics',{})
    rows.append({'variant':v,'view':note,'overall_MAE':m.get('overall_MAE'),'masked_MAE':m.get('masked_MAE'),'support_MAE':m.get('support_MAE'),'H12_masked_MAE':m.get('H12_masked_MAE'),'rel_overall':ri.get('overall'),'rel_masked':ri.get('masked'),'rel_support':ri.get('support'),'node_accept_fraction':j.get('node_acceptance_fraction'),'top1_beneficial_hit':diag.get('top1_beneficial_hit_rate'),'top4_beneficial_hit':diag.get('top4_beneficial_hit_rate'),'beneficial_precision':diag.get('beneficial_precision'),'beneficial_recall':diag.get('beneficial_recall'),'beneficial_auroc':diag.get('beneficial_auroc'),'pairwise_rank_acc':diag.get('pairwise_ranking_accuracy'),'mean_support_mass':j.get('mean_support_mass'),'gate_mean':j.get('gate_mean'),'masked_harm_fraction':j.get('masked_harmful_cell_fraction_vs_M0'),'gate_utility_auroc':j.get('gate_utility_auroc_validation_future_labels_only'),'gate_teacher_mae':j.get('gate_teacher_mae_validation_future_labels_only'),'gate_positive_mean':j.get('gate_positive_mean_validation_future_labels_only'),'gate_nonpositive_mean':j.get('gate_nonpositive_mean_validation_future_labels_only'),'activated_harm_fraction':j.get('activated_cell_harm_fraction'),'masked_activated_harm_fraction':j.get('masked_activated_cell_harm_fraction'),'candidate_eligibility':j.get('mean_candidate_eligibility_fraction'),'m4_teacher_top1':diag.get('teacher_top1_agreement'),'m4_pairwise_rank_acc':diag.get('pairwise_utility_rank_accuracy'),'m4_corr_distill':diag.get('mixture_correction_distill_loss'),'m4_mass_mae':diag.get('support_mass_mae'),'run_dir':str(fs[-1].parent)})
    if 'oracle_after_router_threshold' in j:
        m2=j['oracle_after_router_threshold'];ri2=j.get('relative_improvement_after_router_threshold',{})
        rows.append({'variant':v,'view':'after_router_threshold','overall_MAE':m2.get('overall_MAE'),'masked_MAE':m2.get('masked_MAE'),'support_MAE':m2.get('support_MAE'),'H12_masked_MAE':m2.get('H12_masked_MAE'),'rel_overall':ri2.get('overall'),'rel_masked':ri2.get('masked'),'rel_support':ri2.get('support'),'candidate_eligibility':j.get('mean_candidate_eligibility_fraction'),'m4_teacher_top1':diag.get('teacher_top1_agreement'),'m4_pairwise_rank_acc':diag.get('pairwise_utility_rank_accuracy'),'m4_corr_distill':diag.get('mixture_correction_distill_loss'),'m4_mass_mae':diag.get('support_mass_mae'),'run_dir':str(fs[-1].parent)})
df=pd.DataFrame(rows);print(df.to_string(index=False));out=Path(cfg['project']['exp_root'])/f'presure_{a.stage}_summary.csv';df.to_csv(out,index=False);print('wrote',out)
