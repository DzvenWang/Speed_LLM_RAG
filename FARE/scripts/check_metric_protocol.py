#!/usr/bin/env python3
"""Verify the common Base_SupportBench MAPE interface without a dataset cache."""
import json
import _bootstrap  # inserts FARE root; no benchmark code is copied into FARE

from fare.evaluation import _metrics, common_metrics
import numpy as np


def main():
    _, _, protocol, threshold = common_metrics()
    # Zero/near-zero truths are MAPE-ineligible but remain MAE/RMSE-eligible.
    true=np.array([[[0.,1.,2.,10.]]],dtype=np.float64)
    pred=np.array([[[1.,4.,3.,8.]]],dtype=np.float64)
    valid=np.ones_like(true,dtype=bool)
    direct=np.array([[True,False,True,False]])
    metrics=_metrics(pred,true,valid,direct)
    assert metrics['MAPE_count']==2
    assert metrics['count']==4
    assert metrics['MAPE_coverage']==0.5
    assert np.isclose(metrics['MAPE'],35.0)  # 100 * ((1/2 + 2/10) / 2)
    assert np.isclose(metrics['directly_masked_MAPE'],50.0)
    assert metrics['directly_masked_MAPE_count']==1
    assert np.isclose(metrics['H3_MAPE'],metrics['MAPE'])  # one-step synthetic horizon
    print('FARE_COMMON_METRICS_PASS '+json.dumps({
        'metrics_revision':metrics['metrics_revision'],
        'mape_protocol':protocol,
        'mape_min_target_abs':threshold,
        'MAE':metrics['MAE'],'RMSE':metrics['RMSE'],
        'MAPE':metrics['MAPE'],
        'MAPE_count':metrics['MAPE_count'],
        'MAPE_coverage':metrics['MAPE_coverage'],
    }),flush=True)

if __name__=='__main__':
    main()
