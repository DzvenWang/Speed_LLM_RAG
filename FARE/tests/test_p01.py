"""P0.1 isolated synthetic tests; no real benchmark cache needed."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_full import fixture
from fare.diagnostics_p01 import (make_intervention,oracle_alpha_metrics,metric_np,relative_difference,spearman_pairs)

def test_p01_modes_and_no_future_leak():
    cfg,model,memory,batch=fixture(n=12,b=2,m=36)
    # Use a full legitimate historical latent cache, as the real Pilot does.
    memory.latent=torch.randn(len(memory.x),memory.x.shape[-1],32)
    model.eval();inp={key:batch[key] for key in ('x_obs','obs_mask','tod','dow','window_start')}
    with torch.no_grad():
        ctx,r=model.retrieve_for_diagnostics(inp,memory,diagnostics=True)
        assert r['coarse_ids'].shape[-1]==8
        base=model(inp,memory,return_diagnostics=True)
        override=model(inp,memory,retrieval_override=r,return_diagnostics=True)
        assert torch.allclose(base['prediction'],override['prediction'],atol=2e-5)
        assert override['supported_final_hidden'].shape[-1]==model.stla.d
        for name in ('uniform_selected','random_top64','random_global','time_matched'):
            ret,meta=make_intervention(r,name,ctx,model.regions,memory,inp['window_start'],seed=10)
            assert ret['ids'].shape==r['ids'].shape
            assert torch.isfinite(ret['weights']).all()
            assert torch.allclose(ret['weights'].sum(-1),ret['group_valid'].float(),atol=1e-5)
            assert not (((inp['window_start'][:,None,None]-memory.starts[ret['ids']]).abs()<24)&ret['valid']).any()
            a=model(inp,memory,retrieval_override=ret,return_diagnostics=True)
            assert torch.isfinite(a['prediction']).all()
            assert 'recovered_state' in a
            assert meta['fallback_groups']>=0
            if name=='random_global':
                state_hybrid={**r,'state_ids':ret['ids']}
                latent_hybrid={**r,'latent_ids':ret['ids']}
                assert torch.isfinite(model(inp,memory,retrieval_override=state_hybrid)['prediction']).all()
                assert torch.isfinite(model(inp,memory,retrieval_override=latent_hybrid)['prediction']).all()
    print('FARE_P01_RETRIEVAL_INTERVENTION_PASS')

def test_p01_utility_and_oracle_guard():
    b=np.array([[[2.,5.,0.]]],dtype=np.float32)
    c=np.array([[[4.,3.,0.]]],dtype=np.float32)
    y=np.array([[[4.,4.,1.]]],dtype=np.float32)
    valid=np.ones_like(b,dtype=bool)
    support=np.array([[[True,False,True]]])
    oracle,alpha=oracle_alpha_metrics(b,c,y,valid,support=support)
    assert oracle['MAE']<=metric_np(b,y,valid)['MAE']
    assert oracle['MAE']>0  # no correction where support is false
    assert spearman_pairs([1.,2.,3.],[10.,20.,30.])>0.99
    assert relative_difference(torch.ones(2),torch.zeros(2))['delta_abs']==1.
    print('FARE_P01_ORACLE_ISOLATION_PASS')
