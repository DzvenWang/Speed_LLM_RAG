import numpy as np, torch
from online_presure.memory import MatureEvidenceMemory, MatureEvidenceCase
from online_presure.retriever import retrieve, nodewise_retrieve
from online_presure.evidence import build_nodewise_completion


def main():
    rng=np.random.default_rng(10); H,N,D=12,6,8; mem=MatureEvidenceMemory(32)
    qx=rng.normal(size=(H,N)).astype(np.float32); qobs=np.ones((H,N),bool); miss=np.zeros((H,N),bool)
    miss[:,1]=True; miss[6:,4]=True; qobs[miss]=False; qx[miss]=0.0
    keys=rng.normal(size=(20,D)).astype(np.float32); keys/=np.linalg.norm(keys,axis=1,keepdims=True)+1e-8
    for i in range(20):
        x=qx+rng.normal(scale=.08,size=(H,N)).astype(np.float32); obs=np.ones((H,N),bool)
        if i%4==0: obs[:,1]=False
        if i%5==0: obs[6:,4]=False
        mem.add(MatureEvidenceCase(i,i*5,keys[i],i%288,i%7,x,obs,rng.normal(size=(12,N)).astype(np.float32),np.ones((12,N),bool)))
    cfg={'slots_per_day':288,'recency_tau_steps':288.,'tod_weight':.7,'dow_weight':.3,'latent_weight':.35,'observed_match_weight':.25,
         'complementarity_weight':.20,'recency_weight':.15,'context_weight':.05,'top_k':8,'prefilter_k':20,'observed_match_scale':.5,'min_overlap_fraction':.1,
         'nodewise':{'top_k_per_node':4,'latent_weight':.15,'global_observed_match_weight':.10,'local_observed_match_weight':.30,'complementarity_weight':.25,
                     'recency_weight':.15,'context_weight':.05,'local_observed_match_scale':.5,'min_local_overlap_steps':1}}
    cc={'target':'artificial_missing_only','min_node_overlap_steps':2,'min_support_cases_per_entry':1,'min_weighted_coverage':.1,'consensus_scale_norm':.5,
        'max_gate':.45,'score_temperature':.15}
    pool=retrieve(mem,keys[3],200,3,3,qx,qobs,miss,cfg,top_k=16)
    nr=nodewise_retrieve(mem,pool,qx,qobs,miss,cfg,cc,rng=rng,random_support=False)
    assert 1 in nr['nodes'] and 4 in nr['nodes']
    item={'x_obs':torch.from_numpy(qx),'history_observed':torch.from_numpy(qobs),'artificial_missing':torch.from_numpy(miss)}
    c=build_nodewise_completion(item,mem.rows(),nr,cc)
    assert c['fill_mask'][:,1].any() and c['fill_mask'][6:,4].any()
    assert np.allclose(c['x_aug'][qobs],qx[qobs])
    print('COMPOSITIONAL_SMOKE PASS coverage=',c['coverage'],'nodes=',len(nr['nodes']))
if __name__=='__main__': main()
