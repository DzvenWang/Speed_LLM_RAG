from __future__ import annotations
import numpy as np
import torch

from srd.patch_layout import extract_patch_layout
from srd.patch_transport import candidate_proposal, inject_patch, weighted_topk_proposal


class DummyModel:
    node_num = 6
    # 8 reordered slots = 2 patches * (sps=2 * factors=2); two padding slots.
    reo_all_idx = torch.tensor([0, 1, 2, 2, 3, 4, 5, 5])
    reo_parts_idx = torch.tensor([0, 1, 2, 4, 5, 6])
    ori_parts_idx = torch.tensor([0, 1, 2, 3, 4, 5])
    sps = 2
    factors = 2


def test_patch_layout_exact_partition():
    layout = extract_patch_layout(DummyModel(), {"model_config": {"sps": 2, "factors": 2}})
    assert layout.patch_count == 2
    assert [x.tolist() for x in layout.groups] == [[0, 1, 2], [3, 4, 5]]
    assert sorted(np.concatenate(layout.groups).tolist()) == list(range(6))


def _query_candidate():
    qx = np.array([
        [1, 0, 3, 4],
        [2, 0, 4, 5],
        [3, 0, 5, 6],
    ], np.float32)
    xf = qx.copy(); xf[:, 1] = [2, 3, 4]
    obs = np.ones_like(qx, bool); obs[:, 1] = False
    q = {
        "x_obs": qx, "x_full": xf, "history_observed": obs,
        "restorable": ~obs, "full_valid": np.ones_like(obs),
    }
    cx = np.array([
        [10, 11, 12, 13],
        [11, 12, 13, 14],
        [12, 13, 14, 15],
    ], np.float32)
    c = {"x_full": cx, "full_valid": np.ones_like(obs)}
    return q, c


def test_transport_never_overwrites_observed():
    q, c = _query_candidate()
    cfg = {"patch_transport": {"alignment_scope": "all", "affine_min_points": 2, "affine_scale_clip": [0.5, 2.0], "eps": 1e-6}}
    for method in ["raw_copy", "delta_anchor", "affine_alignment"]:
        p, f = candidate_proposal(q, c, np.array([0, 1]), 3, method, cfg)
        x = inject_patch(q["x_obs"], p, f)
        assert np.allclose(x[q["history_observed"]], q["x_obs"][q["history_observed"]])
        assert not np.any(f & q["history_observed"])


def test_weighted_topk_only_fills_legal_cells():
    q, c = _query_candidate()
    c2 = {"x_full": c["x_full"] + 1, "full_valid": c["full_valid"]}
    cfg = {"patch_transport": {
        "alignment_scope": "all", "affine_min_points": 2, "affine_scale_clip": [0.5, 2.0],
        "eps": 1e-6, "weighted_temperature": 0.35,
    }}
    p, f = weighted_topk_proposal(q, [c, c2], [0.1, 0.2], np.array([0, 1]), 3, cfg, "affine_alignment")
    assert np.any(f[:, 1])
    assert not np.any(f[:, 2:])
    assert np.allclose(p[q["history_observed"]], q["x_obs"][q["history_observed"]])


def test_retrieval_is_invariant_to_hidden_query_truth():
    from srd.patch_retrieval import retrieve_patch

    class Train:
        pass

    tr = Train()
    # Four train windows, T=3, N=2. Candidate 1 is the clear observed-state match.
    tr.x_full_norm = np.array([
        [[0., 0.], [0., 0.], [0., 0.]],
        [[1., 7.], [2., 8.], [3., 9.]],
        [[5., 1.], [6., 2.], [7., 3.]],
        [[2., 3.], [3., 4.], [4., 5.]],
    ], np.float32)
    tr.history_observed = np.ones_like(tr.x_full_norm, bool)
    tr.artificial_missing = np.zeros_like(tr.x_full_norm, bool)
    tr.tod_hist = np.tile(np.array([[0, 1, 2]], np.int64), (4, 1))
    tr.dow_hist = np.zeros((4, 3), np.int64)

    qobs = np.array([[1., 0.], [2., 0.], [3., 0.]], np.float32)
    mask = np.array([[1, 0], [1, 0], [1, 0]], bool)
    q1 = {
        "x_obs": qobs.copy(), "history_observed": mask.copy(),
        "x_full": np.array([[1., 100.], [2., 200.], [3., 300.]], np.float32),
        "tod_hist": np.array([0, 1, 2], np.int64), "dow_hist": np.zeros(3, np.int64),
    }
    q2 = dict(q1)
    q2["x_full"] = np.array([[1., -1000.], [2., -2000.], [3., -3000.]], np.float32)
    cfg = {"patch_retrieval": {
        "min_query_observed_cells": 2, "distance_chunk_size": 2, "min_observed_overlap": 0.5,
        "level_weight": 1.0, "delta_weight": 0.0, "tod_penalty_weight": 0.0,
        "dow_penalty_weight": 0.0, "tod_period": 288, "top_k": 3,
    }}
    ids = np.arange(4, dtype=np.int64)
    r1 = retrieve_patch(q1, tr, ids, np.array([0, 1]), cfg)
    r2 = retrieve_patch(q2, tr, ids, np.array([0, 1]), cfg)
    assert r1.candidate_ids.tolist() == r2.candidate_ids.tolist()
    assert np.allclose(r1.score, r2.score)
    assert int(r1.candidate_ids[0]) == 1


def test_vendor_sd_profile_fallback(tmp_path):
    import torch
    class M:
        node_num = 6
        reo_all_idx = torch.tensor([0, 1, 2, 2, 3, 4, 5, 5])
        reo_parts_idx = torch.tensor([0, 1, 2, 4, 5, 6])
        ori_parts_idx = torch.tensor([0, 1, 2, 3, 4, 5])
    conf = tmp_path / "vendor" / "PatchSTG" / "config"
    conf.mkdir(parents=True)
    (conf / "SD.conf").write_text("spn=4\nfactors=2\nsps=2\n", encoding="utf-8")
    layout = extract_patch_layout(M(), {"model_config": {}}, backbone_root=tmp_path)
    assert layout.patch_size_slots == 4
    assert layout.patch_count == 2
    assert layout.metadata["vendor_profile_patch_count"] == 2


def test_explicit_patch_count_override():
    import torch
    class M:
        node_num = 6
        reo_all_idx = torch.tensor([0, 1, 2, 2, 3, 4, 5, 5])
        reo_parts_idx = torch.tensor([0, 1, 2, 4, 5, 6])
        ori_parts_idx = torch.tensor([0, 1, 2, 3, 4, 5])
    layout = extract_patch_layout(M(), {"model_config": {}}, patch_count_override=2)
    assert layout.patch_size_slots == 4
    assert layout.patch_count == 2


def test_forecast_utility_selectors_respect_k_and_abstain():
    from srd.forecast_utility import PatchAction, select_best_action_by_patch, best_single_action
    base = 2.0
    actions = [
        PatchAction(0, 1, 10, 1.8, 0.2),
        PatchAction(0, 3, 11, 1.5, 0.5),
        PatchAction(1, 1, 20, 2.1, -0.1),
        PatchAction(1, 2, 21, 1.9, 0.1),
    ]
    b1 = select_best_action_by_patch(actions, 1, base, allow_abstain=True)
    assert set(b1) == {0}
    assert b1[0].candidate_id == 10
    b3 = select_best_action_by_patch(actions, 3, base, allow_abstain=True)
    assert b3[0].candidate_id == 11
    assert b3[1].candidate_id == 21
    s1 = best_single_action(actions, 1, base, allow_abstain=True)
    assert s1 is not None and s1.candidate_id == 10
    s3 = best_single_action(actions, 3, base, allow_abstain=True)
    assert s3 is not None and s3.candidate_id == 11


def test_same_episode_pool_is_inference_rank_based_and_capped():
    from srd.forecast_utility import build_same_episode_pool
    import numpy as np
    ids = np.array([[[10, 11, 12], [10, 20, 21], [30, 10, 31]]], dtype=np.int64)
    scores = np.array([[[0.1, 0.2, 0.3], [0.2, 0.4, 0.5], [0.1, 0.2, 0.8]]], dtype=np.float32)
    eligible = np.array([[True, True, True]])
    pool = build_same_episode_pool(ids, scores, eligible, 0)
    assert pool[0]["candidate_id"] == 10
    assert pool[0]["patch_hits"] == 3
    assert pool[0]["coverage"] == 1.0
    capped = build_same_episode_pool(ids, scores, eligible, 0, max_candidates=2)
    assert len(capped) == 2
