from pathlib import Path
import sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from online_pstg_rag.memory import RecentResidualMemory, MemoryCase
from online_pstg_rag.retriever import retrieve
from online_pstg_rag.fusion import fuse_residual


def main():
    H,N,D=12,5,4; mem=RecentResidualMemory(10)
    for i in range(6):
        k=np.array([1,0,0,0],np.float32) if i<3 else np.array([0,1,0,0],np.float32)
        k=k/(np.linalg.norm(k)+1e-8)
        r=np.full((H,N),0.5 if i<3 else -0.5,np.float32)
        mem.add(MemoryCase(i,100+i,k,r,np.ones((H,N),bool),10,2))
    cfg={'slots_per_day':288,'recency_tau_steps':288,'tod_weight':.7,'dow_weight':.3,'latent_weight':.7,'context_weight':.15,'recency_weight':.15,
         'top_k':3,'softmax_temperature':.12,'alpha':.35,'correction_scope':'direct_mask','correction_clip_sigma':1.0,'use_confidence_gate':False,
         'similarity_gate_center':.88,'similarity_gate_scale':.04,'disagreement_scale_raw':3.0}
    a=mem.arrays(); ret=retrieve(a,np.array([1,0,0,0],np.float32),120,10,2,cfg)
    pred,info,_=fuse_residual(np.zeros((H,N),np.float32),np.array([1,0,1,0,1],bool),a,ret,cfg,2.0)
    assert len(ret['local'])==3 and np.all(pred[:,[0,2,4]]>0) and np.allclose(pred[:,[1,3]],0)
    print('SMOKE RAG PASS',info)
if __name__=='__main__': main()
