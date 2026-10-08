"""Run the actual standalone gradient-audit CLI main against synthetic in-memory data.

A regression for v1.3.0, where runtime.setup() leaves memory.keys=None and
step 1 attempted inference before ever building the Train-only key index.
"""
from __future__ import annotations
import contextlib
import io
import json
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1]
for path in (SRC, SRC / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_full import fixture
import diagnose_gradients as audit


def test_actual_gradient_audit_initializes_index_before_forward():
    old_setup = audit.setup
    old_loader = audit.loader
    old_to_device = audit.to_device
    old_config = audit.load_config
    old_argv = sys.argv
    counters = []

    def fake_setup(cfg, smoke=False):
        assert smoke is True
        cfg0, model, memory, batch = fixture(n=9, b=2, m=40)
        # setup() intentionally starts with no learned retrieval key index.
        memory.keys = None
        assert memory.keys is None
        calls = []
        actual_refresh = model.refresh

        def checked_refresh(mem):
            actual_refresh(mem)
            assert mem.keys is not None
            calls.append(mem.keys.detach().clone())

        model.refresh = checked_refresh
        counters.append((memory, calls))
        dataset = [{k: v.detach().clone() for k, v in batch.items()} for _ in range(2)]
        # 0 mean / unit std make synthetic future labels match expected normalizations.
        return model, memory, dataset, None, torch.device('cpu'), {'mean': 0., 'std': 1.}

    try:
        audit.setup = fake_setup
        audit.loader = lambda dataset, *args, **kwargs: iter(dataset)
        audit.to_device = lambda batch, device: batch
        audit.load_config = lambda **kwargs: fixture_cfg()
        sys.argv = ['diagnose_gradients.py', '--dataset', 'TraffiDent_Net_06',
                    '--targets', 'speed', 'flow', '--steps', '2']
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            audit.main()
        lines = buf.getvalue().splitlines()
        ready = [json.loads(s.partition(' ')[2]) for s in lines if s.startswith('FARE_GRADIENT_INDEX_READY ')]
        steps = [json.loads(s.partition(' ')[2]) for s in lines if s.startswith('FARE_GRADIENT_AUDIT ')]
        passed = [s for s in lines if s.startswith('FARE_GRADIENT_AUDIT_PASS ')]
        assert [r['target'] for r in ready] == ['speed', 'flow'], lines
        assert len(steps) == 4, lines
        assert [p['target'] for p in steps] == ['speed', 'speed', 'flow', 'flow']
        assert len(passed) == 2, lines
        assert all('steps=2' in p for p in passed)
        assert all(s['retrieval_valid_fraction'] > 0 for s in steps)
        assert all(torch.isfinite(torch.tensor(list(s['gradient_norms'].values()))).all() for s in steps)
        for memory, calls in counters:
            assert len(calls) == 2, 'Must refresh initially and after optimizer step'
            assert memory.keys is not None
        print('FARE_GRADIENT_AUDIT_INITIAL_INDEX_REGRESSION_PASS targets=speed,flow steps=2', flush=True)
    finally:
        audit.setup = old_setup
        audit.loader = old_loader
        audit.to_device = old_to_device
        audit.load_config = old_config
        sys.argv = old_argv


def fixture_cfg():
    # Avoid any real cache/STLA checkpoint dependency in this CLI-level regression.
    from fare.config import load_config
    return load_config(smoke=True)


if __name__ == '__main__':
    test_actual_gradient_audit_initializes_index_before_forward()
