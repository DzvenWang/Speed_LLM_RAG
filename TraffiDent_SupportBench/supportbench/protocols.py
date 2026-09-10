from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from .utils import ensure_dir, write_json


def _budget_windows(sg_dir: Path, split: str, budget: str) -> set[int]:
    return set(map(int,np.load(sg_dir/'windows'/f'{split}_{budget}.npy')))


def _select_views(views: pd.DataFrame, sg_dir: Path, split: str, budget: str, families=None, severities=None, distribution=None, instances=None) -> pd.DataFrame:
    legal=_budget_windows(sg_dir,split,budget)
    d=views[(views.split==split)&views.window_start.astype(int).isin(legal)].copy()
    if families is not None: d=d[d.family.isin(list(families))]
    if severities is not None:
        sev=np.asarray(list(severities),float); d=d[d.severity.astype(float).map(lambda x: bool(np.isclose(x,sev).any()))]
    if distribution is not None: d=d[d.distribution==distribution]
    if instances is not None: d=d[d.instance.astype(int).isin(list(instances))]
    cols=['view_id','view_group_id','subgraph','split','window_start','family','severity','instance','distribution','layout_id','effective_missing_ratio']
    cols=[c for c in cols if c in d.columns]
    return d[cols].sort_values(['window_start','family','severity','instance','distribution']).reset_index(drop=True)


def _write_views(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent); df.to_csv(path,index=False)


def write_default_protocols(sg_dir: Path, cfg: dict) -> None:
    p=ensure_dir(sg_dir/'protocols'); resolved=ensure_dir(p/'resolved')
    views=pd.read_csv(sg_dir/'views'/'view_manifest.csv')
    main=float(cfg['sensing']['main_severity'])
    general=list(cfg['tracks']['general']['families']); severities=list(map(float,cfg['tracks']['general']['severities']))
    main_fams=list(cfg['tracks']['main_support']['families']); predictive=list(cfg['sensing'].get('predictive_families',[]))
    challenge_fams=list(cfg['tracks']['support_challenge']['source_families'])
    reasoning_eval=list(dict.fromkeys(main_fams+predictive))

    # Track A: broad/general benchmark.
    write_json(p/'track_general.json',{
        'purpose':'General incomplete-sensing benchmark for broad baseline comparability',
        'query_split':'test','families':general,'severities':severities,'distribution':'id','instance':0,
        'metrics':['MAE','RMSE','H3_MAE','H6_MAE','H12_MAE','directly_masked_MAE','support_dependent_MAE'],
        'resolved_files':['resolved/track_general_development.csv','resolved/track_general_full.csv'],
    })
    for budget in ['development','full']:
        _write_views(resolved/f'track_general_{budget}.csv',_select_views(views,sg_dir,'test',budget,general,severities,'id',[0]))

    # Track B: main support-oriented experiment.
    write_json(p/'track_main_support.json',{
        'purpose':'Primary paper setting where structured sensing removes temporal/spatial/propagation context',
        'query_split':'test','families':main_fams,'severity':main,'distribution':'id','instance':0,
        'metrics':['MAE','RMSE','H3_MAE','H6_MAE','H12_MAE','directly_masked_MAE','support_dependent_MAE'],
        'resolved_files':['resolved/track_main_support_development.csv','resolved/track_main_support_full.csv'],
    })
    for budget in ['development','full']:
        _write_views(resolved/f'track_main_support_{budget}.csv',_select_views(views,sg_dir,'test',budget,main_fams,[main],'id',[0]))

    # Explicit support-oriented structured masks: still future-free, based only on Train predictive dependency.
    write_json(p/'track_predictive_structured.json',{
        'purpose':'Support-oriented structured sensing generated from Train-only predictive dependency statistics',
        'query_split':'test','families':predictive,'severity':main,'distribution':'id','instance':0,
        'metrics':['MAE','RMSE','H12_MAE','support_dependent_MAE'],
        'resolved_files':['resolved/track_predictive_structured_development.csv','resolved/track_predictive_structured_full.csv'],
    })
    for budget in ['development','full']:
        _write_views(resolved/f'track_predictive_structured_{budget}.csv',_select_views(views,sg_dir,'test',budget,predictive,[main],'id',[0]))

    # OOD layout stress at matched severity.
    ood_fams=['regional_outage','directional_outage']+predictive
    write_json(p/'track_ood_support.json',{
        'purpose':'Unseen structured sensing-layout stress test at matched severity','query_split':'test','families':ood_fams,'severity':main,'distribution':'ood','instance':0,
        'resolved_files':['resolved/track_ood_support_development.csv','resolved/track_ood_support_full.csv'],
    })
    for budget in ['development','full']:
        _write_views(resolved/f'track_ood_support_{budget}.csv',_select_views(views,sg_dir,'test',budget,ood_fams,[main],'ood',[0]))

    # Track C source lists. The final High-Support-Opportunity subset is derived later from
    # Train-only dependency + current mask + retrieval scores, calibrated only on Validation.
    challenge=cfg['tracks']['support_challenge']
    write_json(p/'track_support_challenge.json',{
        'purpose':'Future-free challenge protocol for RAG/LLM mechanism validation',
        'definition':'SO-v2 Stage A uses network-mean predictive-context loss, support-dependent scope and Train-memory coverage; final SO-v2 is calibrated on Validation realised safe headroom after the numerical backbone is frozen',
        'source_families':challenge_fams,'severity':main,'weights':{
            'context_loss':float(challenge['context_loss_weight']),'retrieval_ambiguity':float(challenge['retrieval_ambiguity_weight']),'memory_coverage':float(challenge['memory_coverage_weight'])},
        'high_quantile':float(challenge['high_quantile']),'low_quantile':float(challenge['low_quantile']),
        'resolved_source_files':['resolved/challenge_val_development.csv','resolved/challenge_test_development.csv','resolved/challenge_val_full.csv','resolved/challenge_test_full.csv'],
        'derived_output':'build_support_opportunity_features.py creates future-free features; fit_support_opportunity_calibrator.py fits Validation outcome calibration and applies the frozen calibrator to Test.',
    })
    for budget in ['development','full']:
        for split in ['val','test']:
            _write_views(resolved/f'challenge_{split}_{budget}.csv',_select_views(views,sg_dir,split,budget,challenge_fams,[main],'id',[0]))

    # GAP 1 paired complete/masked retrieval.
    write_json(p/'gap1_support_retrieval.json',{
        'question':'Does incomplete sensing reduce retrieval of beneficial predictive support, especially in Main/Challenge settings?',
        'paired_complete_and_masked':True,'families':reasoning_eval,'severity':main,'candidate_k':int(cfg['retrieval']['canonical_candidate_k']),
        'metrics':['BeneficialRecall@K','FutureCompatibleRecall@K','OracleBestCandidateError'],
        'resolved_files':['resolved/gap1_pairs_development.csv','resolved/gap1_pairs_full.csv'],
    })
    for budget in ['development','full']:
        legal=_budget_windows(sg_dir,'test',budget)
        comp=views[(views.split=='test')&(views.family=='complete')&views.window_start.astype(int).isin(legal)]
        cmap={int(r.window_start):str(r.view_id) for _,r in comp.iterrows()}
        masked=_select_views(views,sg_dir,'test',budget,reasoning_eval,[main],'id',[0,1])
        pairs=masked.copy(); pairs.insert(0,'complete_view_id',[cmap[int(x)] for x in pairs.window_start]); pairs.rename(columns={'view_id':'masked_view_id'},inplace=True)
        _write_views(resolved/f'gap1_pairs_{budget}.csv',pairs)

    # GAP 2 matched reasoner controls.
    write_json(p/'gap2_support_reasoning.json',{
        'question':'Can Qwen judge predictive support beyond similarity and non-LLM learned controls?',
        'fixed_candidate_pool':True,'candidate_k':int(cfg['retrieval']['reasoner_candidate_k']),
        'controls':['similarity','rules','mlp','small_transformer','qwen_0.5b','qwen_1.5b'],
        'metrics':['BeneficialAUROC','PairwiseRankingAccuracy','HarmfulSupportRate','MAE','RMSE'],
        'train_queries':'all non-complete development Train views','eval_queries':'support-oriented development Val/Test views at main severity',
        'resolved_files':['resolved/gap2_train_development.csv','resolved/gap2_val_development.csv','resolved/gap2_test_development.csv'],
    })
    train_legal=_budget_windows(sg_dir,'train','development')
    trainq=views[(views.split=='train')&(views.family!='complete')&views.window_start.astype(int).isin(train_legal)]
    cols=[c for c in ['view_id','view_group_id','subgraph','split','window_start','family','severity','instance','distribution','layout_id','effective_missing_ratio'] if c in trainq.columns]
    _write_views(resolved/'gap2_train_development.csv',trainq[cols].sort_values(['window_start','family']))
    for split in ['val','test']:
        _write_views(resolved/f'gap2_{split}_development.csv',_select_views(views,sg_dir,split,'development',reasoning_eval,[main],'id',[0]))

    write_json(p/'gap3_grounded_explanation.json',{
        'question':'Are support explanations grounded and contrastively faithful to the same evidence used for support judgement?',
        'metrics':['GroundingAccuracy','DecisionFaithfulness','CounterfactualConsistency','ContrastivePreferenceConsistency'],
        'query_source':'same support-oriented fixed-candidate records as GAP2; no reference prose stored in canonical benchmark',
        'resolved_files':['resolved/gap3_test_development.csv'],
    })
    _write_views(resolved/'gap3_test_development.csv',_select_views(views,sg_dir,'test','development',reasoning_eval,[main],'id',[0]))

    write_json(p/'ablation_retrieval.json',{
        'factors':['single_view_vs_multi_view','temporal_vs_spatial_vs_propagation','episode_vs_local','same_node_vs_network_wide','candidate_k','memory_stride_6_12_24','development_vs_full_memory']})
    write_json(p/'ablation_reasoner.json',{'controls':['similarity','rules','mlp','small_transformer','qwen_0.5b','qwen_1.5b'],'same_candidate_pool':True,'hard_negative_types':list(cfg['reasoning'].get('hard_negative_types',[]))})
    write_json(p/'ablation_fusion.json',{'factors':['backbone_residual','raw_future','fixed_gate','learned_gate'],'primary':'backbone_residual with one alpha selected on Validation only'})
