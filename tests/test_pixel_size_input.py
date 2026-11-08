import contextlib
import io
import unittest
import numpy as np
from bbs_calib.config import validate_config, initial_K, pixel_sizes
from bbs_calib.synthetic import fixture_config, true_scene
from bbs_calib.optimization import calibrate
from bbs_calib.reporting import calibration_document
from test_calibration import make_observations


class PixelSizeInputTests(unittest.TestCase):
    def test_scalar_normalizes_for_downstream_and_snapshot(self):
        for value in (3.69, 5):
            c=fixture_config();c['camera']['pixel_size_um']=value
            validate_config(c)
            self.assertEqual(c['camera']['pixel_size_um'],[float(value)]*2)
            validate_config(c)
            self.assertEqual(c['camera']['pixel_size_um'],[float(value)]*2)

    def test_pair_retains_anisotropy(self):
        c=fixture_config();c['camera']['K_init']=None
        c['camera']['pixel_size_um']=3.69
        square=initial_K(c)
        c['camera']['pixel_size_um']=[3.69,7.38]
        validate_config(c)
        np.testing.assert_allclose(initial_K(c),np.diag([1.,.5])@square)
        self.assertEqual(c['camera']['pixel_size_um'],[3.69,7.38])

    def test_scalar_matches_explicit_pair_with_signed_magnification(self):
        for m in (-100,100):
            c=fixture_config();c['camera'].update(K_init=None,m_px_per_mm=m,pixel_size_um=3.69)
            actual=initial_K(c)
            c['camera']['pixel_size_um']=[3.69,3.69]
            np.testing.assert_array_equal(actual,initial_K(c))

    def test_invalid_values_have_actionable_error(self):
        for value in (None,0,-1,True,'3.69','3.69,3.69',float('nan'),float('inf'),[],[3.69],
                      [1,2,3],[1,-1],[1,float('inf')],[True,3.69],['3.69',3.69],[[1,2]],{}):
            with self.subTest(value=value):
                c=fixture_config();c['camera']['pixel_size_um']=value
                with self.assertRaisesRegex(ValueError,'pixel_size_um.*x=y'):validate_config(c)

    def test_scalar_fit_and_saved_pair(self):
        c=fixture_config();state,poses=true_scene(c);state['kappa']=np.zeros(2)
        c['camera'].update(K_init=(state['K']*1.005).tolist(),omega_p_deg=np.degrees(state['omega']).tolist(),
                           pixel_size_um=3.69,q_scale_mm=None,radius_limit_mm=None)
        c['optimization'].update(release_omega=[False]*3,release_radial=[False]*2,multistart=1,
                                 max_nfev=100,loss='linear',tolerance=1e-11)
        c['validation']['frames']=[]
        obs=make_observations(c,state,poses,5)
        with contextlib.redirect_stdout(io.StringIO()):result=calibrate(c,obs)
        self.assertLess(np.max(np.abs(result['raw_errors'])),1e-5)
        cal=calibration_document(result,c)
        self.assertEqual(cal['config_snapshot']['camera']['pixel_size_um'],[3.69,3.69])
