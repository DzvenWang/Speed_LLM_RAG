from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from brcd.retrieval import build_embedding, search_topk, aggregate_future, oracle_future
from brcd.metrics import summary, fuse, paired_bootstrap_gain
from brcd.backbone_bridge import select_prediction_rows, assert_prediction_alignment

rng = np.random.default_rng(10)
Qtr, Qq, H, P, N = 120, 20, 12, 12, 16
xtr = rng.normal(size=(Qtr,H,N)).astype(np.float32)
xq = rng.normal(size=(Qq,H,N)).astype(np.float32)
mtr = rng.random((Qtr,H,N)) > .4
mq = rng.random((Qq,H,N)) > .4
xtr[~mtr] = 0; xq[~mq] = 0
ytr = rng.normal(size=(Qtr,P,N)).astype(np.float32)
yq = rng.normal(size=(Qq,P,N)).astype(np.float32)
validtr = np.ones_like(ytr, dtype=bool); validq = np.ones_like(yq, dtype=bool)
direct = rng.random((Qq,N)) > .5  # real Base_SupportBench contract: [Q,N]
todtr = rng.integers(0,288,size=(Qtr,H)); dowtr = rng.integers(0,7,size=(Qtr,H))
todq = rng.integers(0,288,size=(Qq,H)); dowq = rng.integers(0,7,size=(Qq,H))
# Disable narrow temporal pool fallback concerns by K=5.
e1 = build_embedding(xtr,mtr); e2 = build_embedding(xq,mq)
top = search_topk(e1,e2,todtr,dowtr,todq,dowq,kmax=5,tod_radius=1,batch_queries=8)
pred,_ = aggregate_future(top,ytr,validtr,k=5,tau=.5,chunk_size=4)
assert pred.shape == yq.shape
oracle,_,chosen = oracle_future(top,ytr,validtr,yq,validq,k=5,chunk_size=4)
assert oracle.shape == yq.shape and chosen.shape == (Qq,)
oracle_m,_,chosen_m = oracle_future(top,ytr,validtr,yq,validq,k=5,extra_mask=direct,chunk_size=4)
assert oracle_m.shape == yq.shape and chosen_m.shape == (Qq,)
base = yq + rng.normal(scale=.2,size=yq.shape).astype(np.float32)
fused = fuse(base,pred,.1)
s = summary(fused,yq,validq,direct)
assert np.isfinite(s['MAE'])
ci = paired_bootstrap_gain(np.mean(np.abs(base-yq),axis=(1,2)),np.mean(np.abs(fused-yq),axis=(1,2)),100,10)

# Regression: real Formal prediction artifacts are full-sized while bounded smoke
# runs request only the first/subset validation rows.  Alignment must happen by
# view_id rather than by requiring equal row counts.
full_pred = rng.normal(size=(50,P,N)).astype(np.float32)
full_ids = np.asarray([f"v{i:03d}" for i in range(50)])
expected = np.asarray(["v007", "v002", "v041", "v010"])
sub_pred, sub_ids = select_prediction_rows(full_pred, full_ids, expected, "synthetic/id_subset")
assert sub_pred.shape[0] == len(expected) and np.array_equal(sub_ids, expected)
assert np.allclose(sub_pred[0], full_pred[7]) and np.allclose(sub_pred[2], full_pred[41])
assert_prediction_alignment(sub_pred, sub_ids, expected, "synthetic/id_subset")

prefix_pred, prefix_ids = select_prediction_rows(full_pred, None, full_ids[:4], "synthetic/prefix", allow_prefix_without_ids=True)
assert prefix_pred.shape[0] == 4 and prefix_ids is None

print("SYNTHETIC_SMOKE_PASS", s, ci, flush=True)
