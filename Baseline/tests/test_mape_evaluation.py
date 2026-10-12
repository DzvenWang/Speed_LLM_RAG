"""Baseline public evaluation-file integration using real NPZ/CSV fixtures."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parent/'Base_SupportBench'))
from baseline.evaluation import evaluate_prediction_file
from experiment_common.predictions import save_prediction_npz

class BaselineMapeEvaluationTests(unittest.TestCase):
    def test_prediction_evaluation_file_and_family(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            pd.DataFrame({'view_id':['a','b'],'family':['entry_random','sensor_outage']}).to_csv(root/'test_manifest.csv',index=False)
            y=np.full((2,12,2),10.,dtype=np.float32)
            y[0,:,0]=0.
            x=y+5.
            valid=np.ones_like(y,dtype=np.uint8)
            direct=np.array([[1,0],[0,1]],dtype=np.uint8)
            for field,arr in [('y_raw',y),('future_valid',valid),('direct_mask',direct)]:
                np.save(root/f'test_{field}.npy',arr)
            # Reverse the prediction order; API must realign by view_id.
            pred_path=root/'p.npz'
            save_prediction_npz(pred_path,['b','a'],x[[1,0]])
            df=evaluate_prediction_file(root,'test',pred_path)
            allr=df[df.family=='ALL'].iloc[0]
            self.assertAlmostEqual(allr['overall_MAE'],5.)
            self.assertAlmostEqual(allr['overall_MAPE'],50.)
            self.assertEqual(allr['overall_MAPE_count'],36)
            self.assertAlmostEqual(allr['overall_MAPE_coverage'],36/48)
            self.assertIn('H12_masked_MAPE', df.columns)
            self.assertIn('masked_MAPE', df.columns)
            self.assertTrue(np.isnan(df[df.family=='entry_random'].iloc[0]['masked_MAPE']))

if __name__=='__main__': unittest.main()
