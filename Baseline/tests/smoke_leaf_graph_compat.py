#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.compat.leaf_graph_ops import install_leaf_graph_compat, symmetric_norm_adj


def unit_contract() -> None:
    # Exact D^-1/2 A D^-1/2 check on a small graph.
    A = torch.tensor([[[1.0, 1.0], [1.0, 3.0]]], dtype=torch.float32)
    d = A.sum(-1)
    expected = d.pow(-0.5).unsqueeze(-1) * A * d.pow(-0.5).unsqueeze(-2)
    got = symmetric_norm_adj(A)
    assert torch.allclose(got, expected, atol=1e-7), (got, expected)

    graph_stub = SimpleNamespace()
    args = SimpleNamespace()
    adj = np.eye(4, dtype=np.float32)
    report = install_leaf_graph_compat(graph_stub, args, adj)
    assert graph_stub.norm_adj is symmetric_norm_adj
    assert len(args.predefined_adjs) == 2
    assert np.array_equal(args.predefined_adjs[0], adj)
    assert np.array_equal(args.predefined_adjs[1], adj)
    assert report['vendor_modified'] is False


def frozen_vendor_integration() -> None:
    # Instantiate both frozen official branches without modifying vendor files.
    vendor = ROOT / 'vendor' / 'LEAF'
    sys.path.insert(0, str(vendor))
    saved = list(sys.argv)
    sys.argv = ['leaf_graph_compat_smoke', '--dataset', 'PEMS03']
    try:
        import cfg as leaf_cfg
        a = leaf_cfg.args
        a.device = torch.device('cpu')
        a.num_nodes = 4
        a.in_dim = 1
        a.out_dim = 1
        a.seq_in_len = 12
        a.seq_out_len = 12
        a.hidden_dim = 8
        a.dropout = 0.0
        a.stgcn_num_layers = 1
        a.hgnn_num_backbone_layers = 1
        a.hgnn_num_head_layers = 1
        a.hgnn_num_hyper_edge = 4
        a.adj = np.eye(4, dtype=np.float32)

        import model.graph as leaf_graph
        report = install_leaf_graph_compat(leaf_graph, a, a.adj)
        from model.graph import GraphBranch
        from model.hypergraph import HypergraphBranch

        graph = GraphBranch()
        hyper = HypergraphBranch()
        batch = {
            'x': torch.randn(2, 12, 4, 1),
            'time_of_day': torch.zeros(2, 12, 4, dtype=torch.long),
            'day_of_week': torch.zeros(2, 12, 4, dtype=torch.long),
        }
        assert tuple(graph(batch).shape) == (2, 12, 4, 1)
        assert tuple(hyper(batch).shape) == (2, 12, 4, 1)
        assert report['predefined_adjs_count'] == 2
    finally:
        sys.argv = saved


def source_contract() -> None:
    # Compatibility belongs in the bridge; frozen source remains untouched.
    graph_text = (ROOT / 'vendor/LEAF/model/graph.py').read_text()
    hyper_text = (ROOT / 'vendor/LEAF/model/hypergraph.py').read_text()
    assert 'norm_adj(' in graph_text
    assert 'args.predefined_adjs' in hyper_text
    runner = (ROOT / 'baseline/runners/leaf_runner.py').read_text()
    assert 'install_leaf_graph_compat' in runner
    assert 'LEAF GRAPH COMPAT=' in runner


def main() -> None:
    unit_contract()
    frozen_vendor_integration()
    source_contract()
    print('LEAF GRAPH COMPAT SMOKE PASS')


if __name__ == '__main__':
    main()
