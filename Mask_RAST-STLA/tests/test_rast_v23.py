import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from mask_rast_stla.config import load_config
from mask_rast_stla.model import MaskRASTSTLA
from mask_rast_stla.memory import build_memory_index
from mask_rast_stla.data import FullTrainRepository
from mask_rast_stla.seasonal import seasonal_innovation,train_future_reference
from mask_rast_stla.trainer import compute_alignment,losses
from mask_rast_stla.wrong_evidence import legal_wrong_indices
from test_rast_v2 import STLA


def make_repo(tmp_path,M=35,H=12,P=12,N=7):
    rng=np.random.default_rng(5)
    np.save(tmp_path/'train_x_full_norm.npy',rng.normal(size=(M,H,N)).astype('f4'))
    np.save(tmp_path/'train_y_norm.npy',rng.normal(size=(M,P,N)).astype('f4'))
    np.save(tmp_path/'train_tod_hist.npy',np.zeros((M,H),dtype='i8'))
    np.save(tmp_path/'train_dow_hist.npy',np.zeros((M,H),dtype='i8'))
    pd.DataFrame({'window_start':np.arange(M)*30}).to_csv(tmp_path/'train_manifest.csv',index=False)
    return FullTrainRepository(tmp_path)


def test_future_innovation_h1_and_alignment(tmp_path):
    torch.manual_seed(6);B,H,P,N=3,12,12,7
    repo=make_repo(tmp_path,N=N)
    cfg=load_config();cfg['retrieval']['top_k']=2;cfg['retrieval']['regional_top_k']=2
    cfg['retrieval']['graph_neighbors']=3
    model=MaskRASTSTLA(STLA(),N,H,P,cfg)
    model.set_graph(torch.tensor([[ (i+j+1)%N for j in range(3)] for i in range(N)]))
    model.set_future_reference(train_future_reference(repo,P,N))
    index=build_memory_index(model.memory_encoder,repo,'cpu',12)
    x=torch.from_numpy(np.array(repo.x[:B],copy=True));mask=torch.ones(B,H,N)
    mask[:,:,0]=0. # Completely missing sensor zero.
    tod=torch.zeros(B,H,dtype=torch.long)
    b=dict(x=x,mask=mask,tod=tod,dow=tod.clone(),window_start=torch.tensor([0,30,60]))
    out=model(b,index,repo)
    assert out['candidate_residuals'].shape[-2:]==(P,N)
    assert bool(out['candidate_residuals'][:,:,0,0].abs().max()>1e-5), 'Fully missing H1 may not be forced to zero'
    assert torch.isfinite(out['prediction']).all()
    b.update(y_norm=torch.randn(B,P,N),y_raw=torch.randn(B,P,N),future_valid=torch.ones(B,P,N,dtype=torch.bool))
    align=compute_alignment(model,b,repo,'cpu')
    loss,_=losses(out,b,0.,1.,cfg,'full',align)
    assert torch.isfinite(align) and torch.isfinite(loss)
    loss.backward()
    assert model.query_encoder.temporal[0].weight.grad is not None
    assert all(p.grad is None for p in model.memory_encoder.parameters())


def test_amp_support_and_seasonal_reference():
    y=torch.randn(2,3,12,5)
    bins=torch.zeros((2,3,5),dtype=torch.long)
    ref=torch.zeros(168,12,5)
    ref[0]=.5
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        out=seasonal_innovation(y,bins,ref)
    assert torch.allclose(out,y-.5,atol=1e-5)
    starts=np.arange(30)*30
    a=legal_wrong_indices([60,90],4,starts,24,'cpu')
    b=torch.cat([legal_wrong_indices([60],4,starts,24,'cpu'),legal_wrong_indices([90],4,starts,24,'cpu')])
    assert torch.equal(a,b)

def test_direct_masked_audit_h12():
    from scripts.evaluate_retrieval import record
    y=np.zeros((2,12,3),dtype=np.float32)
    direct=np.array([[True,False,True],[False,True,False]])
    valid=np.ones_like(y,dtype=bool)&direct[:,None,:]
    row=record('Audit',90,'direct_masked',11,None,y,valid,y,y,y,y)
    assert row['observations']==3 and row['horizon']==12
