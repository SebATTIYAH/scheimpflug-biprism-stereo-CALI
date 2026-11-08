import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from bbs_calib.prism_workflow import pair_complete,refinement_settings,find_grid
from bbs_calib.detection import refine_ellipse
from bbs_calib.step_cli import main
from bbs_calib.io import read_json,read_csv


class SmallGridTests(unittest.TestCase):
    def test_strict_pair(self):
        pair=[dict(view=v,point_id=str(i)) for v in ('left','right') for i in range(108)]
        self.assertTrue(pair_complete(pair));self.assertFalse(pair_complete(pair[:-1]))
        self.assertFalse(pair_complete(pair+[pair[0]]))
        bad=[dict(r) for r in pair];bad[-1]['point_id']='106';self.assertFalse(pair_complete(bad))

    def test_small_circle_subpixel_and_search_radius(self):
        from scipy.special import ndtr
        y,x=np.mgrid[:30,:30];g=np.rint(20+200*ndtr((np.hypot(x-14.2,y-13.7)-3.5)/.65)).astype('uint8')
        cs,_=cv2.findContours((g<120).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        settings=refinement_settings(cs[0],'small')
        self.assertLess(settings['edge_search_px'],2.)
        f=refine_ellipse(g,cs[0],settings)
        np.testing.assert_allclose(f['center'],[14.2,13.7],atol=.12)
        self.assertEqual(refinement_settings(cs[0],'legacy')['edge_search_px'],10)

    def test_grid_area_profile(self):
        g=np.full((310,250),220,np.uint8)
        for i in range(12):
            for j in range(9):cv2.circle(g,(35+j*22,30+i*22),4,20,-1)
        self.assertIsNone(find_grid(g)[1])
        self.assertEqual(len(find_grid(g,min_area_px=8)[1]),108)

    def test_cli_forwarding(self):
        with patch('bbs_calib.workflow_cli.main',return_value=0) as run:
            self.assertEqual(main(['extract','--mode','prism','--images','in','--output','out','--dot-profile','small','--require-complete-pair']),0)
            self.assertIn('--require-complete-pair',run.call_args.args[0])
            self.assertIn('small',run.call_args.args[0])

    def test_incomplete_right_discards_both_views(self):
        from bbs_calib.prism_workflow import extract
        contour=cv2.ellipse2Poly((25,25),(5,5),0,0,360,10).reshape(-1,1,2)
        blobs=[(np.array([25,25]),contour)]*108
        fit=dict(center=[25,25],fit_error=.1,angular_coverage=1.,fit_converged=True)
        with tempfile.TemporaryDirectory() as tmp,patch('bbs_calib.prism_workflow.discover_frames',return_value=[dict(image_id='frame',left='left',right='right')]),patch('bbs_calib.prism_workflow.read_gray',return_value=np.zeros((70,70),np.uint8)),patch('bbs_calib.prism_workflow.find_grid',side_effect=[(blobs,list(range(108)),[]),(None,None,[])]),patch('bbs_calib.prism_workflow.refine_ellipse',return_value=fit):
            out=Path(tmp)/'out';extract(out,0,directory='unused',require_complete_pair=True)
            self.assertEqual(read_csv(out/'observations.csv'),[])
            summary=read_json(out/'extraction_summary.json')
            self.assertEqual(summary['discarded_pairs'],1);self.assertEqual(summary['retained_pairs'],0)
            self.assertFalse(summary['source_images_deleted'])
            self.assertEqual(read_json(out/'audit.json')[0]['views']['left']['refined'],108)
