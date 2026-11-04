"""Regression: target-specific Base_SupportBench dataset root on Gadi."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pcft_stla.v6.data import open_bench


def put_dataset(repo: Path, target: str, dataset='TraffiDent_Net_06', base=None):
    base = base or repo/'data/processed/traffident/base_supportbench'/target
    (base/dataset).mkdir(parents=True, exist_ok=True)
    (base/dataset/'manifest.json').write_text(
        json.dumps({'primary_channel': target}), encoding='utf8'
    )
    return base


@pytest.fixture(autouse=True)
def fake_bench(monkeypatch):
    from supportbench import dataset as module

    class FakeBench:
        def __init__(self, dataset, root):
            self.sg_dir = Path(root) / dataset
            self.channel_names = ['flow', 'occupancy', 'speed']
            self.root = Path(root)

    monkeypatch.setattr(module, 'SupportBench', FakeBench)
    monkeypatch.delenv('PCFT_BENCHMARK_ROOT', raising=False)
    monkeypatch.delenv('TRAFFIDENT_TARGET', raising=False)


def test_gadi_canonical_target_is_separate_for_speed_and_flow(tmp_path):
    speed = put_dataset(tmp_path, 'speed')
    flow = put_dataset(tmp_path, 'flow')
    assert open_bench(tmp_path, 'TraffiDent_Net_06', 'speed').root == speed
    assert open_bench(tmp_path, 'TraffiDent_Net_06', 'flow').root == flow


def test_explicit_parent_and_target_override(tmp_path, monkeypatch):
    parent = tmp_path/'elsewhere'/'benchmark'
    flow = put_dataset(tmp_path, 'flow', base=parent/'flow')
    monkeypatch.setenv('PCFT_BENCHMARK_ROOT', str(parent))
    assert open_bench(tmp_path, 'TraffiDent_Net_06', 'flow').root == flow
    monkeypatch.setenv('PCFT_BENCHMARK_ROOT', str(flow))
    assert open_bench(tmp_path, 'TraffiDent_Net_06', 'flow').root == flow
    monkeypatch.setenv('PCFT_BENCHMARK_ROOT', str(flow/'TraffiDent_Net_06'))
    assert open_bench(tmp_path, 'TraffiDent_Net_06', 'flow').root == flow


def test_reject_cross_target_source_even_when_found(tmp_path):
    flow = put_dataset(tmp_path, 'flow')
    (flow/'TraffiDent_Net_06'/'manifest.json').write_text(
        json.dumps({'primary_channel': 'speed'}), encoding='utf8')
    with pytest.raises(RuntimeError, match='target mismatch'):
        open_bench(tmp_path, 'TraffiDent_Net_06', 'flow')


def test_not_built_fails_instead_of_using_shared_prediction_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match='existing ExperimentCache alone is not sufficient'):
        open_bench(tmp_path, 'TraffiDent_Net_06', 'speed')


def test_target_not_inferred_unless_provided(tmp_path):
    put_dataset(tmp_path, 'speed')
    with pytest.raises(ValueError, match='requires target'):
        open_bench(tmp_path, 'TraffiDent_Net_06')
