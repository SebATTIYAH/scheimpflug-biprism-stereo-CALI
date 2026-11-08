import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
from scipy.special import ndtr
from bbs_calib.blur_filter import profile_width,edge_width,keep_decision
from bbs_calib.step_cli import main
from bbs_calib.blur_filter import filter_dataset
from bbs_calib.io import write_csv,write_json,read_csv,save_image


class BlurFilterTests(unittest.TestCase):
    def test_gaussian_step_width_and_contrast_invariance(self):
        t=np.linspace(-10,10,101)
        for sigma in (1.,2.):
            expected=2.563103*np.sqrt(sigma**2+.5**2)
            self.assertAlmostEqual(profile_width(30+150*ndtr(t/sigma),t),expected,delta=.05)
            self.assertAlmostEqual(profile_width(50+80*ndtr(t/sigma),t),expected,delta=.05)

    def test_low_contrast_unknown(self):
        t=np.linspace(-10,10,101)
        self.assertIsNone(profile_width(100+5*ndtr(t),t))

    def test_threshold_boundary_and_invalid(self):
        self.assertEqual(keep_decision({'edge_width_p75_px':4},4),'kept')
        self.assertEqual(keep_decision({'edge_width_p75_px':4.01},4),'blurred')
        for x in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):keep_decision({'edge_width_p75_px':1},x)

    def test_synthetic_blur_order_no_image_mutation(self):
        y,x=np.mgrid[:100,:110];r=np.hypot(x-53.2,y-48.7)
        scores=[]
        for sigma in (1.,2.5):
            raw=np.rint(30+150*ndtr((r-17)/sigma)).astype('uint8');old=raw.copy()
            contours,_=cv2.findContours((raw<105).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            scores.append(edge_width(raw,contours[0],[53.2,48.7])['edge_width_p75_px'])
            np.testing.assert_array_equal(raw,old)
        self.assertLess(scores[0],4);self.assertGreater(scores[1],4)

    def test_wrong_correspondence_rejected(self):
        contour=cv2.ellipse2Poly((50,50),(15,15),0,0,360,5).reshape(-1,1,2)
        with self.assertRaisesRegex(ValueError,'correspondence'):edge_width(np.zeros((100,100),np.uint8),contour,[60,60])

    def test_cli_popup_and_explicit_paths(self):
        with patch('bbs_calib.step_cli.resolve_mode',return_value='prism'),patch('bbs_calib.step_cli.choose_directory',side_effect=['in','out']) as dialog,patch('bbs_calib.blur_filter.filter_dataset',return_value=('out',{'counts':{},'fit_ready':True})) as run:
            self.assertEqual(main(['filter-blur']),0);self.assertEqual(dialog.call_count,2)
            run.assert_called_once_with('in','out',8.)
        with patch('bbs_calib.step_cli.resolve_mode',return_value='prism'),patch('bbs_calib.step_cli.choose_directory') as dialog,patch('bbs_calib.blur_filter.filter_dataset',return_value=('out',{'counts':{},'fit_ready':True})):
            self.assertEqual(main(['filter-blur','--source','in','--output','out']),0);dialog.assert_not_called()

    def test_complete_filter_preserves_inputs_and_clear_partner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'extract';images=root/'images';images.mkdir()
            for v in ('0','1'):save_image(images/('frame_'+v+'.png'),np.full((40,40),100,np.uint8))
            c=dict(images=dict(mode='paired_crops',directory=str(images),pattern='*.png'),board=dict(rows=12,cols=9),detection=dict(min_points_per_view=1))
            write_json(source/'config.json',c)
            rows=[dict(image_id='frame',view=v,point_id=str(i),u_px='20.25',v_px='21.25',source_u_px='20.25',source_v_px='21.25',accepted=True,review_status='confirmed') for v in ('left','right') for i in (0,1)]
            write_csv(source/'observations.csv',rows);before=(source/'observations.csv').read_bytes()
            with patch('bbs_calib.prism_workflow.find_grid',return_value=([(None,None),(None,None)],[0,1],[])),patch('bbs_calib.blur_filter.edge_width',side_effect=[{'edge_width_p75_px':w} for w in [9,5,5,5]]):
                out,summary=filter_dataset(source,root/'out',8)
            self.assertEqual(summary['counts'],{'blurred':1,'kept':3});self.assertTrue(summary['fit_ready'])
            self.assertTrue(summary['inputs_unchanged']);self.assertEqual((source/'observations.csv').read_bytes(),before)
            result=read_csv(out/'observations.csv');self.assertEqual(result[0]['accepted'],'False')
            self.assertEqual(result[2]['accepted'],'True')
            self.assertEqual([(r['u_px'],r['v_px']) for r in result],[('20.25','21.25')]*4)
            self.assertTrue((out/'README.md').is_file())
            with self.assertRaisesRegex(ValueError,'not empty'):filter_dataset(source,out,8)
