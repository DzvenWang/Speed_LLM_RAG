"""Regression tests for teacher legal replacement sampling.

The original FARE 4.0.1 error used choices[empty, 0] where empty is [B,N]
and choices is [B,N,K], raising IndexError under torch 2.8+.
"""
import pytest
import torch
from fare.teacher import sample_legal_replacement


def test_mixed_legal_and_empty_candidate_rows():
    pool = torch.tensor([
        [[False, True, False, True], [False, False, False, False],
         [True, False, False, False]],
        [[False, False, False, False], [False, False, True, False],
         [False, True, False, False]],
    ], dtype=torch.bool)
    torch.manual_seed(10)
    position, valid = sample_legal_replacement(pool)
    assert position.shape == (2, 3)
    assert valid.shape == (2, 3)
    assert torch.equal(valid, pool.any(-1))
    assert torch.all(position[~valid] == 0)
    assert torch.all(pool.gather(-1, position[..., None]).squeeze(-1) == valid)


def test_all_empty_candidate_rows_have_invalid_swap_labels():
    pool = torch.zeros(2, 388, 16, dtype=torch.bool)
    position, valid = sample_legal_replacement(pool)
    assert torch.count_nonzero(position) == 0
    assert not valid.any()


def test_only_legal_candidates_are_sampled():
    pool = torch.zeros(2, 3, 8, dtype=torch.bool)
    pool[..., 7] = True
    position, valid = sample_legal_replacement(pool)
    assert (position == 7).all()
    assert valid.all()


def test_invalid_empty_candidate_dimension_raises():
    with pytest.raises(ValueError):
        sample_legal_replacement(torch.zeros(2, 3, 0, dtype=torch.bool))
