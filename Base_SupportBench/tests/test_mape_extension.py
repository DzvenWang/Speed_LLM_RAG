"""Synthetic CPU-only canonical evaluation / compatibility tests."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_common.metrics import (
    _metric, mae_rmse, metric_bundle, per_query_metrics,
    backbone_metric_summary, MAPE_PROTOCOL,
)
from experiment_common.constants import METRICS_REVISION


class MapeProtocolTests(unittest.TestCase):
    def test_known_percent_and_existing_mae_rmse(self):
        pred=np.asarray([3., 7., 11., 99., 4.])
        true=np.asarray([2., 10., 10., 0., 1.])
        valid=np.asarray([1, 1, 1, 1, 1], dtype=bool)
        d=mae_rmse(pred,true,valid)
        self.assertAlmostEqual(d['MAE'], (1+3+1+99+3)/5)
        self.assertAlmostEqual(d['RMSE'], np.sqrt((1+9+1+9801+9)/5))
        self.assertAlmostEqual(d['MAPE'], 100*((1/2)+(3/10)+(1/10))/3)
        self.assertEqual(d['MAPE_count'],3)
        self.assertAlmostEqual(d['MAPE_coverage'],3/5)
        self.assertAlmostEqual(_metric(pred,true,valid,'mape'),d['MAPE'])

    def test_invalid_values_mask_and_missing_denominator(self):
        y=np.asarray([0., 0.25, 1., np.nan, 4.])
        p=np.asarray([12., 12., 12., 7., np.inf])
        v=np.asarray([1, 1, 1, 1, 1],dtype=bool)
        d=mae_rmse(p,y,v)
        self.assertEqual(d['count'],3)
        self.assertEqual(d['MAPE_count'],0)
        self.assertTrue(np.isnan(d['MAPE']))
        self.assertAlmostEqual(d['MAPE_coverage'],0.)

    def test_micro_aggregation_not_per_query_average(self):
        p=np.full((2,12,2),5.,dtype=np.float64)
        y=np.full_like(p,10.)
        valid=np.ones_like(p,dtype=bool)
        y[0,:,0]=0.0  # one node all excluded from MAPE
        y[1,:,0]=20.0
        direct=np.asarray([[1,0],[0,1]],dtype=bool)
        b=metric_bundle(p,y,valid,direct)
        self.assertAlmostEqual(b['overall_MAPE'],100*(24*(5/10)+12*(15/20))/36)
        self.assertEqual(b['overall_MAPE_count'],36)
        self.assertEqual(b['overall_count'],48)
        self.assertEqual(b['masked_MAPE_count'],12)
        self.assertAlmostEqual(b['masked_MAPE'],50.)
        self.assertAlmostEqual(b['H12_overall_MAPE'],b['overall_MAPE'])
        self.assertEqual(b['H12_overall_MAPE_count'],3)
        self.assertEqual(b['metrics_revision'],METRICS_REVISION)
        self.assertEqual(b['mape_protocol'], MAPE_PROTOCOL)
        bs=backbone_metric_summary(p,y,valid,direct)
        self.assertAlmostEqual(bs['MAPE'],b['overall_MAPE'])
        self.assertAlmostEqual(bs['directly_masked_MAPE'],b['masked_MAPE'])
        queries=per_query_metrics(p,y,valid,direct)
        self.assertEqual(len(queries),2)
        self.assertIn('H12_MAPE',queries[0])
        self.assertIn('directly_masked_MAPE',queries[1])

    def test_degenerate_empty_valid(self):
        p=np.ones((1,12,2));y=p.copy(); v=np.zeros_like(p,dtype=bool);dm=np.ones((1,2),dtype=bool)
        d=metric_bundle(p,y,v,dm)
        self.assertTrue(np.isnan(d['overall_MAPE']))
        self.assertEqual(d['overall_MAPE_count'],0)
        self.assertTrue(np.isnan(d['overall_MAE']))

if __name__=='__main__':
    unittest.main()
