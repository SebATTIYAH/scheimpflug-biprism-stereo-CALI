import unittest
import tempfile
from pathlib import Path
import cv2
import numpy as np
from bbs_calib.io import transform_points,roi_transform,read_gray,save_image
from bbs_calib.detection import detect_roi
from bbs_calib.grid_ordering import order_grid,symmetry_permutations


class DetectionTests(unittest.TestCase):
    def test_roi_coordinate_transform_roundtrip(self):
        points=np.array([[31.2,44.3],[70,80]])
        for turns in range(4):
            for mirror in (False,True):
                T,_=roi_transform([30,40,60,50],turns,mirror)
                q=transform_points(points,T)
                np.testing.assert_allclose(transform_points(q,np.linalg.solve(T,np.eye(3))),points,atol=1e-12)

    def test_subpixel_image_center_and_roi(self):
        center=np.array([63.28,56.67]);ss=6
        yy,xx=np.mgrid[:120*ss,:140*ss]
        r=((xx+.5)/ss-.5-center[0])**2/9**2+((yy+.5)/ss-.5-center[1])**2/7**2
        high=np.where(r<=1,30.,210.).astype(np.float32)
        high=cv2.GaussianBlur(high,(0,0),.9*ss)
        image=cv2.resize(high,(140,120),interpolation=cv2.INTER_AREA)
        image+=np.random.default_rng(51).normal(0,.3,image.shape)
        image=np.clip(image,0,255).astype(np.uint8)
        result=[r for r in detect_roi(image,[20,15,100,90],{'min_area_px':40}) if r['accepted']]
        self.assertEqual(len(result),1)
        self.assertLess(np.linalg.norm(np.array(result[0]['center'])-center),.15)
        self.assertGreater(len(result[0]['boundary']),50)
        self.assertIsNone(result[0]['center_covariance'])

    def test_blank_and_masked(self):
        blank=np.full((100,100),150,np.uint8)
        self.assertEqual(sum(v['accepted'] for v in detect_roi(blank,[0,0,100,100],{})),0)
        cv2.circle(blank,(50,50),10,20,-1)
        found=detect_roi(blank,[0,0,100,100],{},[[40,40,20,20]])
        self.assertTrue(any('ridge_mask' in v['reject_reason'] for v in found))

    def test_grid_ambiguity_anchors_and_missing(self):
        xy=np.array([[40+22*j+3*i,30+25*i] for i in range(4) for j in range(5)],float)
        order=np.random.default_rng(21).permutation(len(xy));points=xy[order]
        self.assertEqual(order_grid(points,4,5)['status'],'needs_review')
        anchors=[dict(point_id=str(i),u_px=xy[i,0],v_px=xy[i,1]) for i in (0,4,15)]
        result=order_grid(points,4,5,anchors)
        self.assertEqual(result['status'],'confirmed')
        for candidate,pid in result['mapping'].items():
            np.testing.assert_allclose(points[candidate],xy[int(pid)])
        self.assertEqual(order_grid(points[:-1],4,5,anchors)['status'],'needs_review')
        bad=[dict(a) for a in anchors];bad[-1]['u_px']+=30
        self.assertEqual(order_grid(points,4,5,bad)['status'],'needs_review')

    def test_16bit_unicode_path(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'cafe\u0301.tiff';image=np.arange(100,dtype=np.uint16).reshape(10,10)*500
            save_image(path,image)
            np.testing.assert_array_equal(read_gray(path),image)

    def test_square_symmetry(self):
        self.assertEqual(len(symmetry_permutations(np.arange(9).reshape(3,3))),8)
