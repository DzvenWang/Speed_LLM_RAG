from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mask_rast_stla.evaluation import metrics, MAPE_PROTOCOL

def test_mape_protocol_edge_cases():
    truth = np.array([[[0.,1.,2.],[4.,8.,3.]]],dtype=np.float32)
    pred = np.array([[[0.,2.,3.],[6.,8.,6.]]],dtype=np.float32)
    valid = np.ones_like(truth,dtype=bool)
    direct = np.array([[False,True,True]],dtype=bool)
    r=metrics(pred,truth,valid,direct)
    # Five valid nonzero truths, but the truth equal to 1 is excluded.
    assert r['MAPE_count']==4
    expected=np.mean([1./2,2./4,0./8,3./3])*100
    assert np.isclose(r['MAPE'],expected)
    assert np.isclose(r['MAPE_coverage'],4/6)
    assert r['directly_masked_MAPE_count']==3
    assert MAPE_PROTOCOL=='raw_target_percent_exclude_abs_le_1_v1'
