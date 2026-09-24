import numpy as np, torch
from online_presure.memory import MatureEvidenceMemory, MatureEvidenceCase
from online_presure.retriever import retrieve
from online_presure.utility import (OnlineUtilityScorer,prepare_candidate_pool,build_patch_candidate_tables,
    select_patch_supports,sample_probe_specs,build_probe_augmented_histories,FEATURE_NAMES)
from online_presure.evidence import build_patchwise_completion


def main():
    rng=np.random.default_rng(10); T,N,D=12,6,8; mem=MatureEvidenceMemory(64)
    qx=rng.normal(size=(T,N)).astype(np.float32); qobs=np.ones((T,N),bool); miss=np.zeros((T,N),bool)
    miss[9:12,1]=True; miss[3:6,4]=True; qobs[miss]=False; qx[miss]=0.0
    keys=rng.normal(size=(32,D)).astype(np.float32); keys/=np.linalg.norm(keys,axis=1,keepdims=True)+1e-8
    for i in range(32):
        x=qx+rng.normal(scale=.12,size=(T,N)).astype(np.float32); obs=np.ones((T,N),bool)
        if i%5==0: obs[9:12,1]=False
        if i%7==0: obs[3:6,4]=False
        mem.add(MatureEvidenceCase(i,i*4,keys[i],i%288,i%7,x,obs,rng.normal(size=(12,N)).astype(np.float32),np.ones((12,N),bool)))
    rc={'slots_per_day':288,'recency_tau_steps':288.,'tod_weight':.7,'dow_weight':.3,'latent_weight':.35,'observed_match_weight':.25,
        'complementarity_weight':.20,'recency_weight':.15,'context_weight':.05,'top_k':8,'prefilter_k':32,'observed_match_scale':.5,'min_overlap_fraction':.1,
        'nodewise':{'local_observed_match_scale':.5}}
    cc={'target':'artificial_missing_only','min_node_overlap_steps':2,'min_support_cases_per_entry':1,'min_weighted_coverage':.1,'consensus_scale_norm':.5,
        'max_gate':.45,'score_temperature':.15}
    uc={'temporal_patch_size':3,'patch_observed_match_scale':.5,'min_patch_overlap_steps':1,'candidate_consensus_scale':.5,'age_scale_steps':288.,
        'top_k_per_patch':2,'similarity_fallback_weights':{'latent':.1,'global_observed':.1,'node_observed':.15,'patch_observed':.25,'patch_complementarity':.25,'recency':.1,'context':.05},
        'hidden_dims':[32,16],'learning_rate':1e-3,'weight_decay':1e-4,'buffer_size':1000,'min_examples':12,'min_class_examples':1,'batch_size':12,
        'train_steps_per_update':2,'train_every_new_labels':12,'grad_clip':5.,'helpful_threshold_pct':.05,'harmful_threshold_pct':-.05,'harmful_penalty':1.5,
        'helpful_prob_min':0.0,'harmful_prob_max':1.0,'utility_score_min':-9.,'abstain_when_no_support':True,'probes_per_query':6,'probe_top_fraction':.5}
    pool=retrieve(mem,keys[3],200,3,3,qx,qobs,miss,rc,top_k=32); prep=prepare_candidate_pool(mem.rows(),pool,qx,qobs,miss,cc)
    tables=build_patch_candidate_tables(prep,uc,rc); assert (1,3) in tables and (4,1) in tables
    item={'x_obs':torch.from_numpy(qx),'history_observed':torch.from_numpy(qobs),'artificial_missing':torch.from_numpy(miss)}
    scorer=OnlineUtilityScorer(uc,torch.device('cpu'),10)
    specs=sample_probe_specs(tables,uc,rng); pb=build_probe_augmented_histories(item,prep,tables,specs); assert pb is not None and pb['features'].shape[1]==len(FEATURE_NAMES)
    # balanced synthetic labels make the scorer ready and trainable
    X=np.tile(pb['features'],(3,1)); U=np.concatenate([np.full(len(pb['features']),-1.,np.float32),np.zeros(len(pb['features']),np.float32),np.full(len(pb['features']),1.,np.float32)])
    scorer.add(X,U); scorer.maybe_train(); assert scorer.ready
    sel=select_patch_supports(prep,tables,scorer,uc,cc,mode='utility'); c=build_patchwise_completion(item,prep,sel,cc)
    assert np.allclose(c['x_aug'][qobs],qx[qobs]) and c['fill_mask'][9:12,1].any() and c['fill_mask'][3:6,4].any()
    print('UTILITY_SMOKE PASS features=',len(FEATURE_NAMES),'tables=',len(tables),'coverage=',c['coverage'],'classes=',scorer.class_counts())
if __name__=='__main__': main()
