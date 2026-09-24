import numpy as np
from online_presure.memory import MatureEvidenceMemory,MatureEvidenceCase
from online_presure.retriever import retrieve

def main():
    rng=np.random.default_rng(10); mem=MatureEvidenceMemory(32); H,N,D=12,6,8
    keys=rng.normal(size=(20,D)).astype(np.float32); keys/=np.linalg.norm(keys,axis=1,keepdims=True)+1e-8
    qx=rng.normal(size=(H,N)).astype(np.float32); qobs=np.ones((H,N),bool); qmissing=np.zeros((H,N),bool); qmissing[:,2]=True; qobs[:,2]=False
    for i in range(20):
        x=qx+rng.normal(scale=.1,size=(H,N)).astype(np.float32); obs=np.ones((H,N),bool)
        if i%3==0: obs[:,2]=False
        mem.add(MatureEvidenceCase(i,i*5,keys[i],i%288,i%7,x,obs,rng.normal(size=(12,N)).astype(np.float32),np.ones((12,N),bool)))
    cfg={'slots_per_day':288,'recency_tau_steps':288.,'tod_weight':.7,'dow_weight':.3,'latent_weight':.35,'observed_match_weight':.25,
         'complementarity_weight':.20,'recency_weight':.15,'context_weight':.05,'top_k':8,'prefilter_k':16,'observed_match_scale':.5,'min_overlap_fraction':.1}
    r=retrieve(mem,keys[3],200,3,3,qx,qobs,qmissing,cfg); assert r is not None and len(r['case_index'])==8
    assert np.all((r['complementarity']>=0)&(r['complementarity']<=1))
    print('RETRIEVAL_SMOKE PASS',r['case_index'][:4].tolist(),float(np.mean(r['complementarity'])))
if __name__=='__main__': main()
