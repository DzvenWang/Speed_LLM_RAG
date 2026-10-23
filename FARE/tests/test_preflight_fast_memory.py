import json
import numpy as np
import pytest
from fare.memory_preflight import inspect_cached_memory


def _sample(tmp_path, n=7, m=3, h=12, p=12):
    folder=tmp_path/'cached'; folder.mkdir()
    types={'x':'float16','mask':'bool','future':'float16','future_valid':'bool',
           'residual':'float16','residual_valid':'bool','failure_delta':'float16',
           'failure_valid':'bool','starts':'int64','origin':'int64'}
    for name,dtype in types.items():
        shape=(m,) if name in ('starts','origin') else (m,h if name in ('x','mask') else p,n)
        np.save(folder/f'{name}.npy',np.zeros(shape,dtype=dtype))
    manifest={'schema_version':6,'format':'fare5_memmap','nodes':n,'episodes':m,
              'fields':list(types),'residual_provenance':'external_oos_paired',
              'dataset':'TraffiDent_Net_06','target':'speed','backbone_sha256':'hash'}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return folder


def test_cached_memory_inspected_without_full_loading(tmp_path,monkeypatch):
    p=_sample(tmp_path)
    import fare.memory as old
    monkeypatch.setattr(old.EpisodeMemory,'load',lambda *a,**k: (_ for _ in ()).throw(AssertionError('FULL MEMORY LOAD')))
    out=inspect_cached_memory(p,nodes=7,history=12,horizon=12,
                              expected={'target':'speed','backbone_sha256':'hash'})
    assert out['episodes']==3


def test_reject_wrong_checkpoint(tmp_path):
    p=_sample(tmp_path)
    with pytest.raises(RuntimeError,match='provenance mismatch'):
        inspect_cached_memory(p,nodes=7,history=12,horizon=12,expected={'backbone_sha256':'wrong'})


def test_reject_bad_shape(tmp_path):
    p=_sample(tmp_path)
    np.save(p/'residual.npy',np.zeros((3,11,7),dtype=np.float16))
    with pytest.raises(RuntimeError,match='shape'):
        inspect_cached_memory(p,nodes=7,history=12,horizon=12)


def test_reject_non_oof(tmp_path):
    p=_sample(tmp_path)
    mf=p/'manifest.json';doc=json.loads(mf.read_text());doc['residual_provenance']='checkpoint_in_sample_NOT_OOF'
    mf.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError,match='paired out-of-fold'):
        inspect_cached_memory(p,nodes=7,history=12,horizon=12)
