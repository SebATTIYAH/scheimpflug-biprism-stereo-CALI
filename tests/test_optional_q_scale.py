import contextlib
import io
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from bbs_calib.config import validate_config, initial_K, initial_state, resolve_q_scale, board_points
from bbs_calib.synthetic import fixture_config, true_scene
from bbs_calib.optimization import calibrate
from bbs_calib.reporting import calibration_document
from bbs_calib.projection import project, object_points
from bbs_calib.reconstruction import reconstruct_points
from bbs_calib.io import write_json, read_json
from test_calibration import make_observations


class OptionalQScaleTests(unittest.TestCase):
    def test_missing_and_null_validate_without_resolution(self):
        for missing in (False, True):
            c=fixture_config();c['camera']['q_scale_mm']=None
            if missing:del c['camera']['q_scale_mm']
            validate_config(c)
            self.assertIsNone(c['camera'].get('q_scale_mm'))
            with self.assertRaisesRegex(ValueError,'training observations'):initial_state(c)

    def test_invalid_explicit_values_rejected(self):
        for key in ('q_scale_mm','radius_limit_mm'):
            for value in (0,-1,True,'3',float('nan'),float('inf'),[3]):
                with self.subTest(key=key,value=value):
                    c=fixture_config();c['camera'][key]=value
                    with self.assertRaisesRegex(ValueError,key):validate_config(c)

    def test_signed_intrinsics_and_full_frame_center(self):
        q=np.array([[3.,4.],[-2.,1.],[0.,0.]])
        for sign in (-1,1):
            c=fixture_config();cam=c['camera']
            cam.update(q_scale_mm=None,radius_limit_mm=None,K_init=(sign*np.array([[50.,0.],[3.,60.]])).tolist())
            pixels=q@initial_K(c).T+cam['c_px']
            info=resolve_q_scale(c,pixels,['f'])
            self.assertAlmostEqual(info['q_scale_mm'],5)
            self.assertAlmostEqual(cam['radius_limit_mm'],10)
            self.assertEqual(info['training_frames'],['f'])
            self.assertAlmostEqual(initial_state(c)['q_scale'],5)
            # Reusing a resolved config must not change the scale with new pixels.
            again=resolve_q_scale(c,pixels*10,['other'])
            self.assertEqual(info,again)

    def test_manual_values_and_radius_preserved(self):
        c=fixture_config();c['camera'].update(q_scale_mm=7,radius_limit_mm=18)
        info=resolve_q_scale(c,np.empty((0,2)),[])
        self.assertEqual(info['source'],'configured')
        self.assertEqual(initial_state(c)['q_scale'],7)
        self.assertEqual(initial_state(c)['radius_limit_mm'],18)

    def test_degenerate_data_rejected(self):
        c=fixture_config();c['camera']['q_scale_mm']=None
        for pixels in ([],[[float('nan'),0]], [c['camera']['c_px']]):
            with self.assertRaises(ValueError):resolve_q_scale(c,pixels,['f'])

    def test_validation_frames_do_not_set_scale(self):
        c=fixture_config();state,poses=true_scene(c)
        obs=make_observations(c,state,poses,5)
        heldout=obs['frames'][-1];c['validation']['frames']=[heldout]
        c['camera'].update(q_scale_mm=None,radius_limit_mm=None)
        pixels=np.array([v['p'] for v in obs['used'] if v['row']['image_id']!=heldout])
        expected=np.linalg.norm(np.linalg.solve(initial_K(c),(pixels-c['camera']['c_px']).T).T,axis=1).max()
        for v in obs['used']:
            if v['row']['image_id']==heldout:v['p']=v['p']*1000
        with patch('bbs_calib.optimization.initial_state',side_effect=RuntimeError('stop before solver')):
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'stop before solver'):
                calibrate(c,obs)
        self.assertAlmostEqual(c['camera']['q_scale_mm'],expected)
        self.assertNotIn(heldout,c['camera']['q_scale_resolution']['training_frames'])

    def test_auto_fit_save_load_and_reconstruct(self):
        c=fixture_config();state,poses=true_scene(c);state['kappa']=np.zeros(2)
        c['camera'].update(K_init=(state['K']*1.005).tolist(),omega_p_deg=np.degrees(state['omega']).tolist(),
                           q_scale_mm=None,radius_limit_mm=None)
        c['optimization'].update(release_omega=[False]*3,release_radial=[False]*2,multistart=1,
                                 max_nfev=100,loss='linear',tolerance=1e-11)
        c['validation']['frames']=["frame_04"]
        obs=make_observations(c,state,poses,5)
        with contextlib.redirect_stdout(io.StringIO()):result=calibrate(c,obs)
        self.assertLess(np.max(np.abs(result['raw_errors'])),1e-5)
        cal=calibration_document(result,c)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'calibration.json';write_json(path,cal);loaded=read_json(path)
        self.assertEqual(loaded['state']['q_scale'],loaded['distortion']['q_scale_mm'])
        self.assertEqual(loaded['state']['q_scale'],loaded['config_snapshot']['camera']['q_scale_mm'])
        self.assertEqual(loaded['distortion']['normalization']['source'],'auto_training_initial_intrinsics')
        self.assertEqual(loaded['radius_limit_mm'],loaded['state']['radius_limit_mm'])
        _,P=board_points(c);X=object_points(P,poses[-1])
        uv=[project(X,c['views'][v]['branch'],state) for v in ('left','right')]
        rec=reconstruct_points(*uv,loaded)
        np.testing.assert_allclose(rec['X_mm'],X,atol=1e-5)

    def test_nonzero_distortion_scale_conversion_round_trip(self):
        c=fixture_config();state,poses=true_scene(c)
        _,P=board_points(c);X=object_points(P,poses[0])
        uv=[project(X,c['views'][v]['branch'],state) for v in ('left','right')]
        c['camera'].update(q_scale_mm=None,radius_limit_mm=None,K_init=state['K'].tolist())
        resolve_q_scale(c,np.vstack(uv),['f'])
        converted=deepcopy(state);new=c['camera']['q_scale_mm'];ratio=new/state['q_scale']
        converted['q_scale']=new;converted['kappa']=state['kappa']*[ratio**2,ratio**4]
        cal=dict(state=converted,radius_limit_mm=c['camera']['radius_limit_mm'],
                 view_branches={v:c['views'][v]['branch'] for v in ('left','right')})
        for v,pixels in zip(('left','right'),uv):
            np.testing.assert_allclose(project(X,c['views'][v]['branch'],converted),pixels,atol=1e-10)
        rec=reconstruct_points(*uv,cal)
        np.testing.assert_allclose(rec['X_mm'],X,atol=1e-9)
