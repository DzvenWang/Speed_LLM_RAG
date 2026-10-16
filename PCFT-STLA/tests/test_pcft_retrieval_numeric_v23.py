"""Regression coverage for epoch-16 EMA/refreshed retrieval numerical failure."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from pcft_stla.memory import LatentMemoryIndex


def memory_index():
    rng = np.random.default_rng(20261012)
    t = rng.normal(size=(20, 8)).astype(np.float32)
    s = rng.normal(size=(20, 8)).astype(np.float32)
    r = rng.normal(size=(20, 3, 8)).astype(np.float32)
    # This can happen when a stale/unstable encoder emits one invalid key.
    t[4, 0] = np.nan
    r[6, 1, 0] = np.inf
    return LatentMemoryIndex(t, s, np.arange(20, dtype=np.int64)*40, r).to('cpu')


def test_global_fp32_sim_and_train_only_temporal_guard_under_autocast():
    index = memory_index()
    q = torch.randn(3, 8, dtype=torch.bfloat16)
    starts = torch.tensor([0, 80, 160])
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        ids, scores, weights, conf = index.search(q, 'temporal', 6, starts, 24)
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all() and torch.isfinite(weights).all()
    assert not (ids == 4).any()  # NaN Train memory key must not be selected.
    for i, start in enumerate(starts.tolist()):
        chosen = index.starts_np[ids[i].numpy()]
        assert np.all((chosen + 24 <= start) | (chosen >= start + 24))
    assert conf.shape == (3, 3)


def test_regional_fp32_sim_and_invalid_region_filter():
    index = memory_index()
    q = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        ids, scores = index.search_regions(q, 4, torch.tensor([0, 80]), 24)
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()
    assert not (ids[:, 1] == 6).any()


def test_true_temporal_exclusion_is_never_bypassed():
    index = LatentMemoryIndex(np.ones((2, 4), dtype='f4'),
                             np.ones((2, 4), dtype='f4'), [0, 1],
                             np.ones((2, 1, 4), dtype='f4')).to('cpu')
    with pytest.raises(RuntimeError, match='temporally nonoverlapping'):
        index.search(torch.randn(1, 4), 'temporal', 1, torch.tensor([0]), 24)


def test_nonfinite_query_is_not_misdiagnosed_as_temporal_overlap():
    index = memory_index()
    q = torch.randn(1, 8)
    q[0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='Non-finite temporal query encoding'):
        index.search(q, 'temporal', 3, torch.tensor([0]), 24)


def test_encoder_and_index_fp32_in_mixed_precision_full_forward(tmp_path):
    import pandas as pd
    from pcft_stla.config import load_config
    from pcft_stla.model import PCFTSTLA
    from pcft_stla.memory import build_memory_index
    from pcft_stla.data import FullTrainRepository
    from pcft_stla.seasonal import train_future_reference

    class FakeSTLA(torch.nn.Module):
        def set_train_phase(self, phase, last_layers=2):
            pass
        def forward(self, x, mask, tod, dow, support=None, reliability=None):
            b, _, n = x.shape
            pred = x[:, -1:, :].expand(b, 12, n).contiguous()
            return pred, None, None

    count, history, horizon, nodes = 32, 12, 12, 7
    rng = np.random.default_rng(77)
    np.save(tmp_path/'train_x_full_norm.npy', rng.normal(size=(count,history,nodes)).astype('f4'))
    np.save(tmp_path/'train_y_norm.npy', rng.normal(size=(count,horizon,nodes)).astype('f4'))
    np.save(tmp_path/'train_tod_hist.npy', np.zeros((count,history),dtype='i8'))
    np.save(tmp_path/'train_dow_hist.npy', np.zeros((count,history),dtype='i8'))
    pd.DataFrame({'window_start':np.arange(count)*30}).to_csv(tmp_path/'train_manifest.csv',index=False)
    repo = FullTrainRepository(tmp_path)
    cfg = load_config()
    cfg['retrieval']['top_k']=2
    cfg['retrieval']['regional_top_k']=2
    cfg['retrieval']['graph_neighbors']=3
    model = PCFTSTLA(FakeSTLA(), nodes, history, horizon, cfg)
    model.set_graph(torch.tensor([[(i+j+1)%nodes for j in range(3)] for i in range(nodes)]))
    model.set_future_reference(train_future_reference(repo,horizon,nodes))
    index = build_memory_index(model.memory_encoder,repo,'cpu',8)
    x,mask,tod,dow=repo.batch(0,2,'cpu')
    batch=dict(x=x,mask=mask,tod=tod,dow=dow,window_start=torch.tensor([0,30]))
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        encoded=model.query_encoder(x,mask,tod,dow)
        result=model(batch,index,repo)
    assert torch.isfinite(result['prediction']).all()
    assert result['query_temporal_keys'].dtype == torch.float32
    # The StateEncoder outside the protected model forward may be BF16; the
    # production forward always runs retrieval encoders with autocast disabled.
    assert encoded['temporal_key'].shape==(2, cfg['encoder']['retrieval_dim'])
    result['prediction'].float().sum().backward()
    assert model.query_encoder.temporal[0].weight.grad is not None
