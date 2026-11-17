"""ReShift model paths, losses, stage parameter sets, stage keys and layout."""
from __future__ import annotations
import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pytest
import torch
import yaml
from helpers import P,N,FakeSTLA,graph,batch
from reshift.model import CoRSTAv2
from reshift.counterfactual.teacher import PairedTeacher
from reshift.training.normal_anchor import normal_anchor_loss
from reshift.training.layout import Layout,stage_keys
from reshift.training.trainer import stages_for,load_model_state

torch.set_num_threads(1)
CFG_DIR=Path(__file__).resolve().parents[2]/'configs/reshift'
TRAIN={'training':{'forecast_rmse_weight':0.05,'normal_residual_weight':0.1,'cf_response_weight':0.06,
                   'cf_error_gain_weight':0.03,'cf_anchor_gain_weight':0.05,'cf_acceptance_weight':0.02,
                   'cf_error_temperature':0.35,'cf_error_rank_weight':0.02,'cf_anchor_rank_weight':0.02,
                   'cf_latent_weight':0.01,'dose_response_weight':0.04,'dose_gain_weight':0.05,
                   'dose_gain_scale':0.1,'arm_gain_weight':0.05,'arm_gain_scale':0.1}}
FULL={'enabled':True,'arms':['cf','ret_res','ret_shift','ret_both'],'use_features':True,'cf_enabled':True}


def make(retrieval=None,stage='anchor_cf',views=2,seed=5):
    torch.manual_seed(seed)
    m=CoRSTAv2(FakeSTLA(),graph(),hidden=16,failure_dim=8,cf_dim=8,residual_dim=8,horizon=P,heads=2,
               prediction_mode='normal_anchored',dose_levels=(0.5,1.0),dose_temperature=0.01,
               consistency_views=views,consistency_drop=0.3,consistency_seed=3,retrieval=retrieval)
    m.set_stage(stage,use_graph=False)
    return m


def evidence(B=2,K=3,seed=1):
    g=torch.Generator().manual_seed(seed)
    return {'sim':torch.rand(B,K,generator=g),'valid_k':torch.ones(B,K,dtype=torch.bool),
            'p0':torch.randn(B,K,P,N,generator=g),'p_full':torch.randn(B,K,P,N,generator=g),
            'y':torch.randn(B,K,P,N,generator=g),'fv':torch.ones(B,K,P,N,dtype=torch.bool)}


def test_head_does_not_change_pre_retrieval_model():
    obs,_=batch()
    a=make(None).eval();b=make(FULL).eval()
    sa={k:v for k,v in a.state_dict().items()};sb={k:v for k,v in b.state_dict().items() if not k.startswith('retrieval.')}
    assert sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)
    with torch.no_grad():
        oa=a(obs,enable_cf=True,enable_controller=True);ob=b(obs,enable_cf=True,enable_controller=True)
    assert torch.equal(oa['prediction'],ob['prediction'])          # anchor_cf: retrieval unused
    assert ob['arm_kinds'] is None


def test_anchor_control_paths_and_zero_init_policy_rejects():
    obs,_=batch();m=make(FULL,stage='anchor_control').eval()
    with torch.no_grad():o=m(obs,enable_cf=True,enable_controller=True,evidence=evidence())
    assert o['arm_corrections'].shape==(2,5,P,N) and o['arm_gain_pred'].shape==(2,5,P,N)
    assert torch.all(o['arm_selected']==-1)                         # zero-init gains -> reject all
    assert torch.allclose(o['prediction'],o['normal_prediction'])
    with pytest.raises(RuntimeError):m(obs,enable_cf=True,enable_controller=True)
    allow=[k=='cf' for k in m.retrieval.arm_kinds]
    with torch.no_grad():o2=m(obs,enable_cf=True,enable_controller=True,arm_mask=allow)   # audit no_retrieval
    assert o2['retrieval_features'] is not None and torch.all(o2['retrieval_features']==0)


def test_anchor_ret_prediction_is_fusion_arm():
    obs,_=batch();m=make(FULL,stage='anchor_ret').eval()
    with torch.no_grad():o=m(obs,enable_cf=True,enable_controller=False,evidence=evidence())
    j=m.retrieval.fusion_eval_index()
    assert m.retrieval.arm_kinds[j]=='ret_both'
    assert torch.allclose(o['prediction'],o['normal_prediction']+o['arm_corrections'][:,j],atol=1e-6)


def _grads(m):
    return {n for n,p in m.named_parameters() if p.grad is not None and p.grad.abs().sum()>0}


def test_stage_parameter_sets_and_losses():
    obs,b=batch()
    m=make(FULL,stage='anchor_ret');teacher=PairedTeacher(m.stla)
    trainable={n for n,p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith('retrieval.') for n in trainable)
    assert not any('policy' in n for n in trainable)
    m.train();o=m(obs,enable_cf=True,enable_controller=False,evidence=evidence())
    t=teacher(b,o['h0'],o['backbone'])
    loss,details=normal_anchor_loss(o,b,t,'anchor_ret',0.,1.,TRAIN)
    assert torch.isfinite(loss) and 'fusion_mae_ret_both' in details
    loss.backward()
    assert _grads(m)<=trainable and any('gate' in n for n in _grads(m))
    m2=make(FULL,stage='anchor_control');m2.train()
    trainable2={n for n,p in m2.named_parameters() if p.requires_grad}
    assert {n for n in trainable2 if n.startswith('retrieval.')}=={n for n in trainable2 if n.startswith('retrieval.policy')}
    o=m2(obs,enable_cf=True,enable_controller=True,evidence=evidence())
    t=teacher(b,o['h0'],o['backbone'])
    loss,details=normal_anchor_loss(o,b,t,'anchor_control',0.,1.,TRAIN)
    assert torch.isfinite(loss) and 'arm_gain' in details
    loss.backward()
    assert any(n.startswith('retrieval.policy') for n in _grads(m2))
    assert not any(n.startswith(('failure','residual.','graph_residual')) for n in _grads(m2))


def test_shared_checkpoint_loading_ignores_only_retrieval_keys():
    src=make(FULL);dst=make({'enabled':True,'arms':['cf'],'use_features':False})
    load_model_state(dst,src.state_dict(),shared=True)
    bad=dict(src.state_dict());bad['bogus.weight']=torch.zeros(1)
    with pytest.raises(RuntimeError):load_model_state(dst,bad,shared=True)
    with pytest.raises(RuntimeError):load_model_state(dst,src.state_dict(),shared=False)


def _cfg(name):
    cfg=yaml.safe_load((CFG_DIR/name).read_text());cfg['experiment'].update(target='speed',dataset='D',seed=10)
    return cfg


UP={'code_sha256':'c','backbone_sha256':'b','cache_path':'p','split_manifest_sha256':{'train':'t','val':'v'},
    'road_graph_sha256':{},'cf_graph_sha256':None,'smoke':False,'limits':{'train':None}}


def test_stage_keys_share_base_stages_across_retrieval_ablations(tmp_path):
    full=_cfg('reshift_full.yaml');ctrl=_cfg('reshift_no_retrieval.yaml');prior=_cfg('reshift_prior.yaml')
    assert stages_for(full)==('normal','anchor_cf','anchor_ret','anchor_control')
    assert stages_for(ctrl)==('normal','anchor_cf','anchor_control')
    assert 'anchor_ret' in stages_for(prior)
    kf=stage_keys(full,UP,stages_for(full));kc=stage_keys(ctrl,UP,stages_for(ctrl))
    for s in ('s0','bank','normal','anchor_cf'):assert kf[s]==kc[s]
    assert kf['anchor_control']!=kc['anchor_control']
    other=copy.deepcopy(full);other['model']['cf_dim']=16
    ko=stage_keys(other,UP,stages_for(other))
    assert ko['s0']==kf['s0'] and ko['normal']!=kf['normal']
    s0=copy.deepcopy(full);s0['s0']['lr']=1e-3
    assert stage_keys(s0,UP,stages_for(s0))['normal']!=kf['normal']
    lay=Layout(tmp_path/'run',tmp_path/'shared',kf)
    assert lay.dir('normal').parent==tmp_path/'shared' and lay.dir('anchor_ret')==tmp_path/'run'/'anchor_ret'
    lock=lay.acquire('normal')
    with pytest.raises(RuntimeError):lay.acquire('normal')
    lock.unlink();assert lay.acquire('anchor_control') is None
