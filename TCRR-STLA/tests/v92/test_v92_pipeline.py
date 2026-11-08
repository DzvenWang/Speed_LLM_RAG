from pathlib import Path
import numpy as np,pandas as pd,torch
from tcrr_stla.v92.retrieval import VisibleCandidateRetriever
from tcrr_stla.v92.prepare import prepare_normal,prepare_view,V9EvidenceDataset,V9PairedDataset
from tcrr_stla.v9.graph import GraphBundle
from tcrr_stla.v92.training import run_training

class FakeTeacher(torch.nn.Module):
    model_dim=8
    def forward(self,x,m,tod,dow):
        z=(x*m).sum(1)/m.sum(1).clamp_min(1)
        return z[:,None].expand(-1,12,-1),z.unsqueeze(-1).expand(-1,-1,8)

def fixture(tmp):
    root=tmp/'cache';root.mkdir();rng=np.random.default_rng(5)
    for split,count,offset in [('train',40,0),('val',12,400)]:
        x=rng.standard_normal((count,12,4)).astype('float32')
        y=rng.standard_normal((count,12,4)).astype('float32')
        observed=np.ones_like(x,dtype=bool);observed[:,2::4,0]=False
        original=~observed
        arrays=dict(x_full_norm=x,x_obs_norm=x*observed,history_observed=observed,
           artificial_missing=original,y_norm=y,y_raw=y,future_valid=np.ones_like(y,bool),
           direct_mask=np.ones_like(y,bool),tod_hist=np.zeros((count,12),np.int64),
           dow_hist=np.zeros((count,12),np.int64))
        for name,array in arrays.items():np.save(root/f'{split}_{name}.npy',array)
        pd.DataFrame({'window_start':np.arange(count)*24+offset*24+100}).to_csv(root/f'{split}_manifest.csv',index=False)
    return root

def test_visible_retrieval_and_identity(tmp_path):
    root=fixture(tmp_path)
    search=VisibleCandidateRetriever(root,pool_size=12,shared=2)
    x=np.load(root/'train_x_obs_norm.npy')[20];m=np.load(root/'train_history_observed.npy')[20]
    ids,scores=search.select(x,m,100+24*20,k=6)
    assert ids.shape==(6,) and scores.shape==(6,)
    assert (ids>=0).all() and (ids<20).all()
    changed_x=x.copy();changed_x[:,1]+=4
    changed_ids,changed_scores=search.select(changed_x,m,100+24*20,k=6)
    assert ids[0]==changed_ids[0] and ids[1]==changed_ids[1]
    assert not np.array_equal(scores,changed_scores)

def test_full_prepare_training_and_replay_same_anchor(tmp_path):
    root=fixture(tmp_path);evidence=tmp_path/'evidence';evidence.mkdir()
    teacher=FakeTeacher();device=torch.device('cpu')
    prepare_normal(root,teacher,device,evidence,limit=40,chunk=8)
    for split,view in [('train',0),('train',1),('val',0)]:
        prepare_view(root,split,view,teacher,device,evidence,None,k=6,chunk=4,
              limit=12 if split=='train' else None,seed=10)
    ds=V9EvidenceDataset(root,'train',(0,1),root=evidence)
    assert len(ds)==24
    pairs=V9PairedDataset(root,evidence)
    assert len(pairs)==12
    a=pairs[8]['official'];b=pairs[8]['paired']
    assert torch.equal(a['candidate_ids'][:4],b['candidate_ids'][:4])
    assert a['cf'].shape==(6,12,4)
    assert not torch.equal(a['mask'],b['mask'])
    assert not torch.equal(a['score'],b['score'])
    # Each view's model query can use only its own mask and visible observations.
    assert not torch.any(b['x'][b['mask']==0])
    cfg=dict(batch_size=4,eval_batch_size=4,workers=0,
       normal_epochs=1,align_epochs=1,utility_epochs=1,joint_epochs=1,paired_epochs=1,
       lr=.001,hidden=12,cf_hidden=8,bound=3.,blocks=3,max_align=1.5,temperature=.2,
       alignment_mode='signed',graph_mode='gnn_reliability',temporal_mode='cnn',null_mode='conditional',
       amp=False,clip=5.,patience=3,utility_weight=.02,rank_weight=.01,
       alignment_weight=.002,marginal_weight=.005,full_info_weight=.01,
       cross_weight=.01,compat_weight=.01)
    out=tmp_path/'run';best=run_training(root,evidence,GraphBundle(torch.eye(4)),8,0.,1.,out,10,cfg,smoke=True)
    assert best.is_file()
    for st in ('normal','align','utility','joint','paired'):
        assert (out/f'best_{st}.pt').is_file()
    hist=__import__('json').loads((out/'history.json').read_text())
    assert len(hist)==5
    assert hist[-1]['train']['matched_candidates']>0

def test_diagnostics_report_on_completed_smoke(tmp_path):
    import json
    from tcrr_stla.v92.diagnostics import diagnose_run
    root=fixture(tmp_path);evidence=tmp_path/'evidence';evidence.mkdir()
    teacher=FakeTeacher();device=torch.device('cpu')
    prepare_normal(root,teacher,device,evidence,limit=40,chunk=8)
    for split,view in [('train',0),('train',1),('val',0)]:
        prepare_view(root,split,view,teacher,device,evidence,None,k=4,chunk=4,
               limit=12 if split=='train' else None,seed=10)
    cfg=dict(batch_size=4,eval_batch_size=4,workers=0,normal_epochs=1,align_epochs=1,
       utility_epochs=1,joint_epochs=1,paired_epochs=1,lr=.001,hidden=12,cf_hidden=8,
       bound=3.,blocks=3,max_align=1.5,temperature=.2,alignment_mode='signed',
       graph_mode='gnn_reliability',temporal_mode='cnn',null_mode='conditional',amp=False,
       clip=5.,patience=3,utility_weight=.02,rank_weight=.01,alignment_weight=.002,
       marginal_weight=.005,full_info_weight=.01,cross_weight=.01,compat_weight=.01)
    run=tmp_path/'run'
    run_training(root,evidence,GraphBundle(torch.eye(4)),8,0.,1.,run,10,cfg,smoke=True)
    identity={'version':'9.2.0','target':'speed','dataset':'TraffiDent_Net_06',
       'seed':10,'formal_cache':str(root),'evidence_root':str(evidence),'smoke':True}
    path=run/'run_identity.json';path.write_text(json.dumps(identity))
    report=diagnose_run(path,max_queries=8,batch_size=4,bootstrap=10)
    assert (run/'diagnostics/report.html').is_file()
    assert (run/'diagnostics/ablation_results.csv').is_file()
    assert report['cohort_rows']==8

def test_strong_official_retrieval_is_preserved_and_pair_is_independent(tmp_path):
    root=fixture(tmp_path);evidence=tmp_path/'evidence';evidence.mkdir()
    teacher=FakeTeacher();dev=torch.device('cpu')
    prepare_normal(root,teacher,dev,evidence,limit=40,chunk=8)
    k=6;ids=np.full((40,k),1,np.int32);scores=np.full((40,k),-.01,np.float32)
    for qi in range(40):
        ids[qi]=np.array([12,11,10,9,8,7])
        scores[qi]=np.array([3,2,1,0,-1,-2])
    shortlist=tmp_path/'retrieval.npz'
    np.savez(shortlist,version=25,indices=ids,scores=scores,coverage=np.ones_like(scores))
    prepare_view(root,'train',0,teacher,dev,evidence,shortlist,k=k,chunk=4,limit=20,seed=10)
    prepare_view(root,'train',1,teacher,dev,evidence,None,k=k,chunk=4,limit=20,seed=10)
    data=V9EvidenceDataset(root,'train',(0,1),root=evidence)
    a,b=data[36],data[37]
    assert torch.equal(a['candidate_ids'][:4],b['candidate_ids'][:4])
    assert (a['candidate_ids'][4:]==12).any() or (a['candidate_ids'][4:]==11).any()
    assert not torch.equal(a['score'],b['score'])
    assert torch.all(a['candidate_ids'][a['valid']]<18)

def test_strict_wrong_mask_replay_diagnostic_same_cohort(tmp_path,monkeypatch):
    import json
    from tcrr_stla.v92.diagnostics import diagnose_run
    from tcrr_stla.v5 import shared_adapter
    import tcrr_stla.config as config_module
    root=fixture(tmp_path);evidence=tmp_path/'evidence';evidence.mkdir()
    teacher=FakeTeacher();device=torch.device('cpu')
    prepare_normal(root,teacher,device,evidence,limit=40,chunk=8)
    for split,view in [('train',0),('train',1),('val',0)]:
        prepare_view(root,split,view,teacher,device,evidence,None,k=4,chunk=4,
                     limit=12 if split=='train' else None,seed=10)
    cfg=dict(batch_size=4,eval_batch_size=4,workers=0,normal_epochs=1,align_epochs=1,
        utility_epochs=1,joint_epochs=1,paired_epochs=1,lr=.001,hidden=12,cf_hidden=8,
        bound=3.,blocks=3,max_align=1.5,temperature=.2,alignment_mode='signed',
        graph_mode='gnn_reliability',temporal_mode='cnn',null_mode='conditional',
        amp=False,clip=5.,patience=3,utility_weight=.02,rank_weight=.01,
        alignment_weight=.002,marginal_weight=.005,full_info_weight=.01,
        cross_weight=.01,compat_weight=.01)
    run=tmp_path/'run'
    run_training(root,evidence,GraphBundle(torch.eye(4)),8,0.,1.,run,10,cfg,smoke=True)
    # Force a CF checkpoint for exercising complete controls, without claiming
    # that this synthetic data actually supports CF gain.
    (run/'selected_checkpoint.json').write_text(json.dumps({'stage':'joint','path':str(run/'best_joint.pt')}))
    identity=dict(version='9.2.0',target='speed',dataset='TraffiDent_Net_06',seed=10,
        formal_cache=str(root),evidence_root=str(evidence),smoke=True,
        backbone_checkpoint=str(tmp_path/'fake_best.pt'),adapter_checkpoint=str(tmp_path/'fake_adapter.pt'))
    path=run/'run_identity.json';path.write_text(json.dumps(identity))
    monkeypatch.setattr(config_module,'load_config',lambda _: {'experiment':{},'project':{'retrieval_root':str(tmp_path)}})
    monkeypatch.setattr(shared_adapter,'initial_model',lambda *args:(FakeTeacher(),{'mean':0.,'std':1.}))
    report=diagnose_run(path,max_queries=8,batch_size=4,bootstrap=12,wrong_mask_queries=4)
    assert report['wrong_mask_replay']['cohort_rows']==4
    assert report['wrong_mask_replay']['cohort_sha256']
    assert 'headroom' in report and report['headroom']['local_oracle_signed_gain_norm'] is not None
    assert (run/'diagnostics/gradient_audit.json').is_file()
