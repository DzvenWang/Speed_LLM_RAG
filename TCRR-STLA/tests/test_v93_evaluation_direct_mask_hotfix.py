"""Regression for the expanded-mask in-place failure in TEST evaluation."""
import importlib.util
from pathlib import Path
import pytest
import torch


def _mask_fn():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_v93.py'
    spec = importlib.util.spec_from_file_location('tcrr_v93_evaluate_hotfix', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.valid_direct_mask


def test_expanded_direct_mask_and_valid_positions():
    fn = _mask_fn()
    direct = torch.tensor([[True, False, True], [False, True, True]])
    valid = torch.tensor([
        [[True, True, False], [True, False, True]],
        [[False, True, True], [True, True, False]],
    ])
    result = fn(direct, valid)
    torch.testing.assert_close(result, direct[:, None, :] & valid)
    assert result.shape == valid.shape
    assert torch.equal(direct, torch.tensor([[True, False, True], [False, True, True]]))
    assert result.is_contiguous()


def test_direct_mask_three_dimensions_and_shape_check():
    fn = _mask_fn()
    direct = torch.tensor([[[True, False], [True, True]]])
    valid = torch.tensor([[[True, True], [False, True]]])
    torch.testing.assert_close(fn(direct, valid), direct & valid)
    with pytest.raises(RuntimeError, match='Wrong direct mask shape'):
        fn(torch.ones(1, 2, 3, dtype=torch.bool), valid)
