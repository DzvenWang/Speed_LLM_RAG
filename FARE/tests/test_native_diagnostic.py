"""Native source integration: synthetic FARE model only, no hidden mocked predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fare.diagnostics_once import native_export
from fare.diagnostics_once.data import load_shard


def test_native_export_real_model_paths(fixture_data,monkeypatch,tmp_path):
    import fare.pipeline as pipeline
    import fare.config as conf
    import fare.protocol as protocol
    model,memory,_=fixture_data
    memory.attach_graph(model.graph)
    families=['entry_random','regional_outage','directional_outage','temporal_outage','sensor_outage']
    class SimpleDataset:
        split='test'
        def __init__(self,*args,**kwargs):
            self.meta=pd.DataFrame({'window_start':[200+i*25 for i in range(5)], 'failure_family':families})
            self.ix=np.arange(5)
        def __len__(self):return 5
        def __getitem__(self,i):
            b=torch.Generator().manual_seed(101+i)
            h=12;n=6;p=12
            x=torch.randn((h,n),generator=b)
            target=x.mean(0)[None].expand(p,n).clone()
            return {'x':x, 'mask':torch.ones(h,n,dtype=torch.bool),
                'tod':torch.zeros(h,dtype=torch.long),'dow':torch.zeros(h,dtype=torch.long),
                'origin':torch.tensor(500+i*30),'row_id':torch.tensor(i),
                'target_raw':target,'future_valid':torch.ones((p,n),dtype=torch.bool)}
    cp=tmp_path/'best.pt';bp=tmp_path/'backbone.pt';bp.touch()
    mem_path=tmp_path/'memory';mem_path.mkdir();(mem_path/'manifest.json').write_text('{}')
    cfg={'experiment':{'target':'speed','dataset':'TraffiDent_Net_06','seed':10},
         'memory':{'purge_steps':0},'training':{'batch_size':2}}
    meta={'backbone_checkpoint':str(bp),'memory_path':str(mem_path),
          'backbone_sha256':'abc','memory_sha256':'def'}
    torch.save({'state_dict':model.state_dict(),'config':cfg,'metadata':meta},cp)
    monkeypatch.setattr(pipeline,'prepare',lambda *args,**kwargs:(model,memory,None,None,torch.device('cpu'),
                    {'backbone_sha256':'abc','memory_sha256':'def'}))
    monkeypatch.setattr(conf,'cache_dir',lambda *args,**kwargs:tmp_path)
    monkeypatch.setattr(protocol,'SupportDataset',SimpleDataset)
    (tmp_path/'test_manifest.csv').write_text('window_start,failure_family\n')
    out=tmp_path/'out'
    native_export.export_once(target='speed',checkpoint=str(cp),output_dir=str(out),
          severities=[40],failure_families=families,max_queries_per_group=1,
          seed=10,split='test',config={'dataset':'TraffiDent_Net_06','allow_exploratory_test':True,
                                      'inference_batch_size':1,'oracle_selected_slots':2})
    for family in families:
        ds=load_shard(out/f'speed_s40_{family}.npz',allowed_splits=('test',))
        assert ds.n==1
        assert ds.candidate_predictions.shape==(1,2,12,6)
        assert ds.metadata['oracle_scope']=='selected_slot_only_not_pool_oracle'
        assert 'no_direction' in ds.modes
        assert 'null_evidence' in ds.modes
        assert np.isfinite(ds.modes['full']).all()
        assert np.allclose(ds.modes['null_evidence'],ds.modes['no_evidence'],atol=1e-5)
