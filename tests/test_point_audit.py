import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from bbs_calib.point_audit import ranked_points, reproduce, ellipse_curve, generate
from bbs_calib.step_cli import main
from bbs_calib.io import write_json


class PointAuditTests(unittest.TestCase):
    def records(self):
        obs=dict(image_id='a',view='right',point_id='2',u_px=1700.,v_px=100.,source_u_px=12.,source_v_px=100.,fit_error_px=.2)
        err=dict(obs,du_px=.3,dv_px=-.4,error_px=.5)
        return [err],[obs],{'a':{'right':[1688,0]}}

    def test_residual_sign_and_crop_origin(self):
        row=ranked_points(*self.records(),1)[0]
        self.assertAlmostEqual(row['predicted_source_u_px'],12.3)
        self.assertAlmostEqual(row['predicted_source_v_px'],99.6)

    def test_bad_coordinates_rejected(self):
        e,o,c=self.records();o[0]['source_u_px']=13
        with self.assertRaisesRegex(ValueError,'coordinates'):ranked_points(e,o,c,1)

    def test_invalid_norm_and_top(self):
        e,o,c=self.records();e[0]['error_px']=.9
        with self.assertRaises(ValueError):ranked_points(e,o,c,1)
        with self.assertRaises(ValueError):ranked_points(*self.records(),0)

    def test_ranking_and_duplicate_protection(self):
        e,o,c=self.records()
        e.append(dict(e[0],point_id='3',du_px=0.,dv_px=.1,error_px=.1))
        o.append(dict(o[0],point_id='3'))
        self.assertEqual([r['point_id'] for r in ranked_points(e,o,c,20)],[2,3])
        with self.assertRaisesRegex(ValueError,'Duplicate'):ranked_points(e+e,o,c,1)

    def test_reproduce_synthetic_circle(self):
        y,x=np.mgrid[:100,:110];radius=np.hypot(x-53.2,y-48.7)
        raw=np.rint(30+150/(1+np.exp(-(radius-16.)/1.))).astype('uint8')
        contours,_=cv2.findContours((raw<105).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        original=raw.copy();crop,fit=reproduce(raw,contours[0])
        np.testing.assert_array_equal(raw,original)
        np.testing.assert_allclose(np.array(fit['center'])+fit['patch_origin_source_px'],[53.2,48.7],atol=.04)
        self.assertEqual(ellipse_curve(fit).shape,(361,2))

    def test_cli_dialogs_and_explicit_paths(self):
        with patch('bbs_calib.step_cli.choose_directory',side_effect=['fit','audit']) as dialog,patch('bbs_calib.point_audit.generate',return_value='audit') as run:
            self.assertEqual(main(['audit-points','--top','5']),0)
            self.assertEqual(dialog.call_count,2);run.assert_called_once_with('fit','audit',5)
        with patch('bbs_calib.step_cli.choose_directory') as dialog,patch('bbs_calib.point_audit.generate',return_value='out'):
            self.assertEqual(main(['audit-points','--calibration','in','--output','out']),0)
            dialog.assert_not_called()

    def test_unsupported_model_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_json(p/'calibration.json',{'model_version':'single'})
            with self.assertRaisesRegex(ValueError,'requires rectangular'):generate(p,p/'out')
            self.assertFalse((p/'out').exists())
