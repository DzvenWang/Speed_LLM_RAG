import numpy as np
from online_presure.arbitrator import OnlineCompletionArbitrator, visible_gate_features, FEATURE_NAMES


def main():
    cfg={'enabled':True,'initial_probability':.85,'warmup_multiplier':1.0,'min_multiplier':0.0,'max_multiplier':1.0,
         'min_examples':8,'buffer_size':32,'batch_size':8,'steps_per_update':2,'learning_rate':1e-2,'weight_decay':1e-4,
         'grad_clip':5.0,'utility_weight':1.0,'utility_scale_pct':3.0,'utility_weight_cap':3.0}
    a=OnlineCompletionArbitrator(cfg,seed=10)
    x=np.linspace(0,1,len(FEATURE_NAMES),dtype=np.float32)
    assert a.multiplier(x)==1.0
    for i in range(24):
        target=1.0 if i%3 else 0.0
        a.add_example(x+0.01*i,target,2.0 if target else -5.0)
    p=a.multiplier(x)
    assert 0.0<=p<=1.0 and a.examples_seen==24 and a.updates>0
    print('ARBITRATOR_SMOKE PASS multiplier=',p,'updates=',a.updates)
if __name__=='__main__': main()
