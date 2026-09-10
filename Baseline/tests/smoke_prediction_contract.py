from pathlib import Path
import tempfile, numpy as np, pandas as pd, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from baseline.evaluation import evaluate_prediction_file
with tempfile.TemporaryDirectory() as td:
    p=Path(td); M,P,N=4,12,3
    pd.DataFrame({'view_id':[f'v{i}' for i in range(M)],'family':['a','a','b','b']}).to_csv(p/'val_manifest.csv',index=False)
    y=np.ones((M,P,N),np.float32); valid=np.ones_like(y,np.uint8); mask=np.ones((M,N),np.uint8)
    np.save(p/'val_y_raw.npy',y); np.save(p/'val_future_valid.npy',valid); np.save(p/'val_direct_mask.npy',mask); np.save(p/'val_support_mask.npy',mask)
    np.savez(p/'pred.npz',view_id=np.array([f'v{i}' for i in range(M)]),pred=y.copy())
    d=evaluate_prediction_file(p,'val',p/'pred.npz'); assert float(d.iloc[0].overall_MAE)==0.0
print('PREDICTION CONTRACT SMOKE PASS')
