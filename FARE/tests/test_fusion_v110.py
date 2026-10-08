"""v1.1.0 correction/calibration policy unit and end-to-end regression checks."""
from __future__ import annotations
import copy,sys,tempfile
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fare.control import UtilityRiskController
from fare.alpha import scalar_sweep,save_sweep
from fare.config import config_from_checkpoint
from fare.losses import losses
from test_full import fixture


def test_policy_is_bounded_and_trainable():
    for horizon in (3,12):
        ctrl=UtilityRiskController(16,horizon,max_alpha=.6,initial_alpha=.12)
        raw=torch.randn(2,horizon,5,8)
        raw[...,7]=torch.arange(horizon).view(1,horizon,1)/max(1,horizon-1)
        U,R,G=ctrl.from_raw(raw)
        assert U.shape==R.shape==G.shape==(2,horizon,5)
        assert torch.allclose(G,torch.full_like(G,.12),atol=1e-6)
        assert (G>=0).all() and (G<=.6).all()
        # Supervise the effective correction directly; verify this is NOT
        # just an inert or decorative gate.
        loss=((G-.35)**2).mean()
        loss.backward()
        assert ctrl.policy[-1].weight.grad is not None
        assert torch.isfinite(ctrl.policy[-1].weight.grad).all()
        assert ctrl.horizon_bias.grad is not None
        opt=torch.optim.AdamW(ctrl.parameters(),lr=.03)
        before=float(G.mean())
        for _ in range(15):
            opt.zero_grad();_,_,g=ctrl.from_raw(raw)
            ((g-.35)**2).mean().backward();opt.step()
        with torch.no_grad():_,_,after_g=ctrl.from_raw(raw)
        assert float(after_g.mean())>before+0.03,(before,float(after_g.mean()))


def test_checkpoint_inference_gate_exactly_matches_forecast():
    cfg,model,memory,batch=fixture(n=13,b=2,m=35)
    model.eval();model.refresh(memory)
    with torch.no_grad():
        initial=model(batch,memory)
        assert torch.allclose(initial['prediction'],initial['base_prediction']+
                              initial['final_gate']*(initial['candidate_prediction']-initial['base_prediction']),atol=1e-5)
        assert initial['final_gate'].max()<=model.controller.max_alpha
        # Regression for v1.0.4: previously calibration could scale past the
        # claimed gate yet still log the unscaled, pre-calibration gate.
        model.calibration_multiplier.fill_(20)
        calibrated=model(batch,memory)
        assert calibrated['final_gate'].max()<=model.controller.max_alpha+1e-6
        assert torch.allclose(calibrated['prediction'],calibrated['base_prediction']+
                              calibrated['final_gate']*(calibrated['candidate_prediction']-calibrated['base_prediction']),atol=1e-5)
        model.calibration_multiplier.fill_(0)
        off=model(batch,memory)
        assert torch.equal(off['prediction'],off['base_prediction'])


def test_candidate_supervision_and_safe_inference():
    cfg,model,memory,batch=fixture(n=10,b=2,m=35)
    model.set_phase('injection');model.train();model.refresh(memory)
    out=model(batch,memory)
    ls=losses(model,batch,out,memory,'injection',0.,1.,cfg)
    assert ls['candidate'].item()>0 and ls['correction_energy'].item()>=0
    ls['total'].backward()
    assert model.controller.policy[-1].weight.grad is not None
    assert torch.isfinite(model.controller.policy[-1].weight.grad).all()
    assert model.innovation.local[-1].weight.grad is not None
    stripped={k:batch[k] for k in ('x_obs','obs_mask','tod','dow','window_start')}
    with torch.no_grad():p=model(stripped,memory)
    assert p['prediction'].shape==(2,12,10)


def test_alpha_sweep_val_and_shape_safety():
    rng=np.random.default_rng(123)
    true=np.zeros((5,12,11),dtype=np.float32)
    base=np.ones_like(true)
    candidate=-4*np.ones_like(true) # alpha=.2 exactly reaches truth
    v=np.ones_like(true,dtype=bool)
    report=scalar_sweep(base,candidate,true,v,alphas=(0,.1,.2,.3,.6),max_alpha=.6)
    assert report['best']['alpha']==.2,report
    assert report['best']['MAE']<1e-6
    with tempfile.TemporaryDirectory() as t:
        save_sweep(report,t)
        assert (Path(t)/'alpha_sweep.csv').is_file()
    bad=base.copy();bad[0,0,0]=np.nan
    try:scalar_sweep(bad,candidate,true,v)
    except FloatingPointError:pass
    else:raise AssertionError('NaN on valid position was ignored')
    try:scalar_sweep(base,candidate,true,v,alphas=(.75,))
    except ValueError:pass
    else:raise AssertionError('Excessive alpha was allowed')


def test_checkpoint_config_recovery():
    cfg,model,memory,batch=fixture()
    cfg['control']['initial_alpha']=.27
    ck={'config':copy.deepcopy(cfg)}
    got=config_from_checkpoint(ck)
    assert got['control']['initial_alpha']==.27
    assert got is not cfg and got['control'] is not cfg['control']
    got['control']['initial_alpha']=.45
    assert cfg['control']['initial_alpha']==.27

if __name__=='__main__':
    for fn in (test_policy_is_bounded_and_trainable,
               test_checkpoint_inference_gate_exactly_matches_forecast,
               test_candidate_supervision_and_safe_inference,
               test_alpha_sweep_val_and_shape_safety,
               test_checkpoint_config_recovery):
        print('RUN',fn.__name__,flush=True)
        fn()
        print('PASS',fn.__name__,flush=True)
    print('FARE_V110_FUSION_TEST_PASS',flush=True)
