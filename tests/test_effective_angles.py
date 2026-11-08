import unittest
import numpy as np
from bbs_calib.geometry import scheimpflug
from bbs_calib.angles import effective_angles
from bbs_calib.prism_workflow import find_grid
import cv2

class EffectiveAngleTests(unittest.TestCase):
    def test_real_grid_extractor_one_to_one(self):
        gray=np.full((850,800),220,np.uint8)
        expected=np.array([[100+55*j+6*i,60+56*i-2*j] for i in range(12) for j in range(9)])
        for xy in expected:cv2.circle(gray,tuple(xy),14,20,-1)
        for i in range(6):cv2.circle(gray,(40,200+i*30),6,20,-1)
        blobs,selected,_=find_grid(gray)
        self.assertIsNotNone(selected);self.assertEqual(len(set(selected)),108)
        np.testing.assert_allclose(np.array([blobs[i][0] for i in selected]),expected,atol=.5)

    def test_known_mapping_and_sign(self):
        for a,b in [(8,5),(5,-.04),(1.4,-1.4),(0,5),(5,0),(0,0)]:
            K=-30*scheimpflug(*np.radians([a,b]));r=effective_angles(K)
            rebuilt=r['m_px_per_mm']*scheimpflug(*np.radians([r['alpha_deg'],r['beta_deg']]))
            np.testing.assert_allclose(rebuilt,K,atol=1e-8)
            self.assertAlmostEqual(r['total_tilt_deg'],np.degrees(np.arccos(np.cos(np.radians(a))*np.cos(np.radians(b)))),places=6)
            self.assertEqual(r['status'],'conditional_effective_angle_not_independently_measured')
