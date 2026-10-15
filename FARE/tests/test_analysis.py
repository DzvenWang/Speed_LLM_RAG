import torch,numpy as np

def test_full_internal_analysis_with_dummy_protocol(fixture_data,tmp_path,monkeypatch):
    from fare import evaluation
    from fare.analysis import analyze,MODES
    from torch.utils.data import Dataset
    model,mem,b=fixture_data
    class Tiny(Dataset):
        def __len__(self):return 2
        def __getitem__(self,i):
            out={k:(v[i] if v.ndim else v) for k,v in b.items()}
            out['target_raw']=out['target'].clone()
            return out
    def local_test_only_bundle(pred,true,valid,direct):
        # THIS IS A TEST DOUBLE; production evaluation *never* uses this implementation.
        diff=(pred-true)[valid]
        return {'overall_MAE':float(np.abs(diff).mean()),'overall_RMSE':float(np.sqrt(np.mean(diff**2)))}
    monkeypatch.setattr(evaluation,'canonical_metric_bundle',lambda:local_test_only_bundle)
    c={'training':{'batch_size':2,'workers':0},'memory':{'purge_steps':0},
       'analysis':{'max_counterfactual_queries':1,'bootstrap_samples':20},'experiment':{'seed':10}}
    records=analyze(model,mem,{'val':Tiny()},torch.device('cpu'),c,tmp_path)
    assert len(records)==len(MODES)+1
    assert (tmp_path/'module_contributions.csv').exists()
    assert (tmp_path/'candidate_counterfactual.csv').exists() or (tmp_path/'counterfactual'/'candidate_counterfactual.csv').exists()
    assert (tmp_path/'research_report.md').exists()
