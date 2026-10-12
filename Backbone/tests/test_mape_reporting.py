"""Ensure MAPE is present in Backbone textual and structured reporting."""
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parent/'Base_SupportBench'))
from backbone.evaluation_reporting import severity_row, scenario_rows, severity_table_lines, scenario_table_lines
from backbone.metrics import backbone_metric_summary

class BackboneMapeReportingTests(unittest.TestCase):
    def test_scenarios_and_severity(self):
        x=np.asarray([[[2.0,4.0] for _ in range(12)]])
        truth=np.asarray([[[4.0,8.0] for _ in range(12)]])
        valid=np.ones_like(x,dtype=bool)
        m=np.asarray([[True, False]])
        stats=backbone_metric_summary(x,truth,valid,m)
        for k,v in list(stats.items()):
            stats[k+'__entry_random']=v
        overall=severity_row(stats,40,10)
        self.assertAlmostEqual(overall['MAPE'],50.)
        self.assertEqual(overall['MAPE_count'],24)
        self.assertAlmostEqual(overall['MAPE_coverage'],1.)
        self.assertEqual(overall['train_data_mode'] if 'train_data_mode' in overall else 'unspecified','unspecified')
        scenario=scenario_rows(stats,['entry_random'],40,10,{'entry_random':1})
        self.assertEqual(len(scenario),1)
        self.assertAlmostEqual(scenario[0]['Masked_MAPE'],50.)
        self.assertIn('MAPE', '\n'.join(severity_table_lines([overall])))
        self.assertIn('MAPE', '\n'.join(scenario_table_lines(scenario)))

if __name__=='__main__': unittest.main()
