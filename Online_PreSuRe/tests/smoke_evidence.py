import numpy as np, torch
from online_presure.memory import MatureEvidenceCase
from online_presure.evidence import build_completion,build_transition,fuse_predictions

def main():
    H,N=12,5; qx=np.zeros((H,N),np.float32); obs=np.ones((H,N),bool); miss=np.zeros((H,N),bool); miss[:,1]=True; obs[:,1]=False
    q={'x_obs':torch.from_numpy(qx),'history_observed':torch.from_numpy(obs),'artificial_missing':torch.from_numpy(miss),
       'y_raw':torch.zeros((12,N)),'direct_mask':torch.tensor([0,1,0,0,0],dtype=torch.bool),'support_mask':torch.tensor([0,0,1,0,0],dtype=torch.bool),
       'future_valid':torch.ones((12,N),dtype=torch.bool)}
    rows=[]
    for i in range(3):
        x=np.zeros((H,N),np.float32); x[:,1]=1.0+.1*i; co=np.ones((H,N),bool); y=np.zeros((12,N),np.float32); y[:,1]=10+i
        rows.append(MatureEvidenceCase(i,i,np.ones(4,np.float32),0,0,x,co,y,np.ones((12,N),bool)))
    r={'local':np.arange(3),'score':np.array([1.,.9,.8],np.float32),'case_index':np.arange(3)}
    cc={'target':'artificial_missing_only','min_node_overlap_steps':2,'min_support_cases_per_entry':1,'min_weighted_coverage':.1,'consensus_scale_norm':.5,'max_gate':.45,'score_temperature':.15}
    tc={'min_node_overlap_steps':2,'min_support_cases_per_entry':1,'min_weighted_coverage':.1,'consensus_scale_raw':8.,'max_gate':.35,'score_temperature':.15}
    c=build_completion(q,rows,r,cc); assert c['fill_mask'][:,1].all() and np.all(c['x_aug'][:,0]==0)
    t=build_transition(q,rows,r,c,1.0,tc); assert t['valid'][:,1].all()
    slow=np.zeros((12,N),np.float32); cp=np.ones((12,N),np.float32)
    f,gc,ge=fuse_predictions(slow,cp,t,c,q,cc,tc,{'scope':'direct_and_support','max_external_weight':.65},True,True)
    assert f.shape==(12,N) and np.allclose(f[:,0],0) and np.max(gc+ge)<=.65001
    print('EVIDENCE_SMOKE PASS fill=',int(c['fill_mask'].sum()),'max_gate=',float(np.max(gc+ge)))
if __name__=='__main__': main()
