#!/usr/bin/env python3
from pathlib import Path
import sys,numpy as np,torch
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.features import build_node_dual_features_batch,row_normalize_adjacency,build_evidence_descriptor,build_backbone_descriptor
from presure.ranker import DualQueryRanker,soft_teacher,ranker_loss
from presure.model import DualQueryConditionalForecaster
from presure.metrics import weighted_forecast_loss

def main():
    rng=np.random.default_rng(10); B=2; K=8; M=16; H=P=4; N=6; C=5
    x=rng.normal(size=(M,H,N)).astype(np.float32); obs=rng.random((M,H,N))>0.2; x*=obs; base=rng.normal(40,2,size=(M,P,N)).astype(np.float32); hidden=rng.normal(size=(M,N,C)).astype(np.float32); tod=np.tile(np.arange(H),(M,1)); dow=np.zeros((M,H),np.int64); adj=row_normalize_adjacency(np.eye(N,k=1)+np.eye(N,k=-1))
    state,ev=build_node_dual_features_batch(x,obs,base,hidden,tod,dow,adj,40,5); assert state.shape[:2]==(M,N) and ev.shape[:2]==(M,N)
    ged=build_evidence_descriptor(x,obs,tod,dow,adj,6,4,10,8); gbd=build_backbone_descriptor(base,hidden,40,5,6,3,20,8); dual=np.concatenate([ged,gbd],1); ci=np.stack([rng.choice(M,K,replace=False) for _ in range(B)]); rank=DualQueryRanker(dual.shape[1],32,64,0.0); q=torch.from_numpy(dual[:B]); c=torch.from_numpy(dual[ci]); dist=torch.from_numpy(rng.random((B,K)).astype(np.float32)); score=rank(q,c); loss=ranker_loss(score,soft_teacher(dist,0.2)); loss.backward(); assert torch.isfinite(loss)
    model=DualQueryConditionalForecaster(state.shape[-1],ev.shape[-1],P,d_model=32,state_hidden=48,evidence_hidden=48,support_hidden=48,decoder_hidden=64,attention_heads=4,dropout=0.0,alpha_init=0.25,attention_batch_chunk=32768)
    qi=np.arange(B); sup=ci; sy=torch.from_numpy((base[sup]+rng.normal(0,1,size=(B,K,P,N))).astype(np.float32)); sv=torch.ones(B,K,P,N,dtype=torch.bool); o=model(torch.from_numpy(state[qi]),torch.from_numpy(ev[qi]),torch.from_numpy(base[qi]),torch.from_numpy(state[sup]),torch.from_numpy(ev[sup]),sy,torch.from_numpy(base[sup]),sv,score.detach(),40,5)
    # zero-init retrieval head guarantees exact protected-backbone start despite alpha=0.25
    assert torch.allclose(o['pred'],torch.from_numpy(base[qi]),atol=1e-5); assert torch.allclose(o['retrieval_pred'],torch.from_numpy(base[qi]),atol=1e-5); assert abs(float(o['alpha'].mean().detach())-0.25)<1e-4
    target=torch.from_numpy((base[qi]+rng.normal(0,1,size=(B,P,N))).astype(np.float32)); valid=torch.ones(B,P,N,dtype=torch.bool); direct=torch.ones(B,N,dtype=torch.bool); support=torch.ones(B,N,dtype=torch.bool); fl,_=weighted_forecast_loss(o['pred'],target,valid,direct,support,{'overall_weight':1,'masked_weight':2.5,'support_weight':0.5}); aux,_=weighted_forecast_loss(o['retrieval_pred'],target,valid,direct,support,{'overall_weight':1,'masked_weight':2.5,'support_weight':0.5}); (fl+0.75*aux).backward(); assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Formal-shape regression test: B*N*P = 16*388*12 = 74496 > CUDA SDPA's 65535 batch limit.
    # The chunked support-attention helper must handle this without changing semantics.
    stress_total=16*388*12; stress_k=2
    with torch.no_grad():
        stress_seq=torch.randn(stress_total,stress_k,model.d)
        stress_pad=torch.zeros(stress_total,stress_k,dtype=torch.bool)
        stress_out=model._support_self_attention_chunked(stress_seq,stress_pad)
        assert stress_out.shape==stress_seq.shape and torch.isfinite(stress_out).all()
    print('SMOKE PreSuRe v1.2 PASS',flush=True)
if __name__=='__main__': main()
