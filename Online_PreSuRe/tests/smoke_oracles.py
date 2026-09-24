import numpy as np, torch
from online_presure.oracles import perfect_completion_item,retrieved_completion_oracles,retrieved_entry_oracles_by_k

def main():
    T,N=4,3
    xo=np.zeros((T,N),np.float32); xf=xo.copy(); xf[:,1]=2.0
    miss=np.zeros((T,N),bool); miss[:,1]=True
    item={'x_obs':torch.from_numpy(xo),'x_full':torch.from_numpy(xf),'artificial_missing':torch.from_numpy(miss)}
    p,m=perfect_completion_item(item); assert np.allclose(p['x_obs'].numpy()[:,1],2.0) and m[:,1].all()
    vals=np.stack([np.ones((T,N),np.float32),np.full((T,N),3.0,np.float32),np.full((T,N),2.1,np.float32),np.full((T,N),2.0,np.float32)])
    avail=np.ones_like(vals,bool); c={'candidate_values':vals,'candidate_available':avail}
    o=retrieved_completion_oracles(item,c); assert o is not None and o['case_fill_count']==T and o['entry_fill_count']==T
    ks=retrieved_entry_oracles_by_k(item,c,[1,2,4]); assert set(ks)=={1,2,4}
    assert ks[4]['recon_mae_norm'] <= ks[2]['recon_mae_norm']+1e-8
    print('ORACLE_SMOKE PASS case=',o['case_recon_mae_norm'],'entry@4=',ks[4]['recon_mae_norm'])
if __name__=='__main__': main()
