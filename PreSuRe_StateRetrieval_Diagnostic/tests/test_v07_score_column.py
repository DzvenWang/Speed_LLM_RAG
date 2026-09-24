import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def _load_selector_script():
    root = Path(__file__).resolve().parents[1]
    path = root / 'scripts' / '08_run_predictive_utility_selector.py'
    spec = importlib.util.spec_from_file_location('v07_selector_script', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _Train:
    def row(self, cid):
        return {'candidate_id': int(cid)}


def test_compose_preserves_underscore_score_column(monkeypatch):
    mod = _load_selector_script()
    selected = pd.DataFrame([
        {
            'candidate_id': 7,
            'patch_id': 0,
            'rank': 2,
            '_score': 0.42,
            'forecast_utility': 0.15,
        }
    ])
    q = {
        'x_obs': np.zeros((2, 2), dtype=np.float32),
        'artificial_missing': np.ones((2, 2), dtype=bool),
    }
    layout = SimpleNamespace(groups=[np.array([0], dtype=np.int64)])

    def fake_candidate_proposal(qexec, cand, group, L, transport, cfg):
        proposal = np.ones((2, 2), dtype=np.float32)
        fill = np.ones((2, 2), dtype=bool)
        return proposal, fill

    def fake_inject_patch(x, proposal, fill):
        out = x.copy()
        out[fill] = proposal[fill]
        return out

    monkeypatch.setattr(mod, 'candidate_proposal', fake_candidate_proposal)
    monkeypatch.setattr(mod, 'inject_patch', fake_inject_patch)

    x, used = mod._compose(q, selected, _Train(), layout, 12, 'affine_alignment', {}, {})
    assert np.all(x == 1.0)
    assert used == [(0, 7, 2, 0.42, 0.15)]
