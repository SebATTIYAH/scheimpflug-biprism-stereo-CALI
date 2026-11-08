import contextlib
import io
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from bbs_calib.standard_grid import find_standard_grid,outer_candidates
from bbs_calib.prism_workflow import refinement_settings
from bbs_calib.detection import refine_ellipse


def board(radius=18,pitch=68,rows=6,cols=6):
    margin=radius+30
    image=np.full((rows*pitch+2*margin,cols*pitch+2*margin),180,np.uint8)
    centers=[]
    for i in range(rows):
        for j in range(cols):
            center=(margin+j*pitch,margin+i*pitch);centers.append(center)
            cv2.circle(image,center,radius,25,-1)
            if (i,j) in ((1,1),(1,4),(3,1)):
                cv2.circle(image,center,max(2,radius//3),180,-1)
    return cv2.GaussianBlur(image,(0,0),.8),np.array(centers,float)


class StandardGridTests(unittest.TestCase):
    def test_ring_holes_are_never_candidates(self):
        binary=np.zeros((160,160),np.uint8)
        cv2.circle(binary,(80,80),40,255,-1);cv2.circle(binary,(80,80),15,0,-1)
        blobs=outer_candidates(binary,1,10,10000)
        self.assertEqual(len(blobs),1);self.assertGreater(blobs[0][2],4500)
        # If the exterior is too large, do not silently substitute the hole.
        self.assertEqual(outer_candidates(binary,1,10,2000),[])

    def test_mixed_ring_and_solid_grids_at_two_scales(self):
        for radius,pitch in ((18,68),(43,155)):
            im,truth=board(radius,pitch)
            blobs,selected,attempts=find_standard_grid(im,6,6)
            self.assertIsNotNone(selected,attempts)
            self.assertEqual(len(set(selected)),36)
            centers=np.array([blobs[i][0] for i in selected])
            np.testing.assert_allclose(centers,truth,atol=1.)
            for pid in (0,7,10,19):
                co=blobs[selected[pid]][1]
                fit=refine_ellipse(im,co,refinement_settings(co,'standard'))
                np.testing.assert_allclose(fit['center'],truth[pid],atol=.3)
                self.assertGreater(min(fit['semi_axes']),radius*.85)
                self.assertLess(fit['fit_error'],.8)
                self.assertGreaterEqual(fit['angular_coverage'],.8)

    def test_missing_dot_is_not_filled(self):
        im,truth=board();x,y=map(int,truth[0]);cv2.rectangle(im,(x-25,y-25),(x+25,y+25),180,-1)
        self.assertIsNone(find_standard_grid(im,6,6)[1])

    def test_square_transpose_and_sign_ambiguity(self):
        from bbs_calib.standard_grid import ordered_grid
        pts=np.array([[50+j*60,50+i*60] for i in range(6) for j in range(6)],np.float32)
        blobs=[(p,None,300.,10.) for p in pts]
        shuffled=pts.reshape(6,6,2).transpose(1,0,2)[:,::-1]*.4
        with patch('bbs_calib.standard_grid.cv2.findCirclesGrid',return_value=(True,shuffled.reshape(-1,1,2))):
            selected,error=ordered_grid(blobs,6,6,3.)
        np.testing.assert_allclose(pts[selected],pts)
        self.assertLess(error,1e-8)

    def test_search_stays_near_outer_rim(self):
        angles=np.linspace(0,2*np.pi,96,endpoint=False)
        co=np.column_stack([30+18*np.cos(angles),30+18*np.sin(angles)]).astype(np.float32).reshape(-1,1,2)
        setting=refinement_settings(co,'standard')
        self.assertAlmostEqual(setting['edge_search_px'],5.4,places=4)
        self.assertEqual(setting['min_profile_contrast'],refinement_settings(None,'legacy')['min_profile_contrast'])

    def test_cli_forwards_standard_and_complete_pair(self):
        from bbs_calib.step_cli import main
        with patch('bbs_calib.workflow_cli.main',return_value=0) as run,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['extract','--mode','prism','--images','in','--output','out',
                '--rows','6','--cols','6','--dot-profile','standard','--require-complete-pair']),0)
            args=run.call_args.args[0]
            self.assertEqual(args[args.index('--dot-profile')+1],'standard')
            self.assertIn('--require-complete-pair',args)

    def test_invalid_size_limits(self):
        im,_=board()
        for lo,hi in ((0,10),(20,10),(1,float('nan'))):
            with self.assertRaises(ValueError):find_standard_grid(im,6,6,min_area_px=lo,max_area_px=hi)


if __name__=='__main__':unittest.main()
