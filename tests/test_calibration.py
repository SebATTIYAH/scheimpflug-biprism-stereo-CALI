import unittest
from pathlib import Path
import tempfile
import numpy as np
from bbs_calib.synthetic import fixture_config,true_scene
from bbs_calib.config import board_points,validate_config
from bbs_calib.optimization import observations,calibrate,matrix_diagnostics,derive_angles,Parameters,Objective,array_data,finite_jacobian
from bbs_calib.projection import project,object_points
from bbs_calib.io import write_csv,read_json


def make_observations(c,state,poses,frames=5):
    ids,P=board_points(c);used=[]
    for i,pose in enumerate(poses[:frames]):
        X=object_points(P,pose)
        for view in ('left','right'):
            s=c['views'][view]['branch'];uv=project(X,s,state)
            for pid,xyz,p in zip(ids,P,uv):
                row=dict(image_id='frame_%02d'%i,view=view,point_id=pid,u_px=p[0],v_px=p[1],accepted=True,review_status='confirmed')
                used.append(dict(row=row,P=xyz,p=p,W=np.eye(2),branch=s))
    return dict(used=used,excluded=[],frames=['frame_%02d'%i for i in range(min(frames,len(poses)))])


class CalibrationTests(unittest.TestCase):
    def test_circle_diameter_optional_for_center_calibration(self):
        c=fixture_config();c['board']['diameter_mm']=None
        validate_config(c,'calibrate');validate_config(c,'detect')
        del c['board']['diameter_mm']
        validate_config(c,'calibrate')
        for invalid in (0,-1,float('nan')):
            c['board']['diameter_mm']=invalid
            with self.assertRaisesRegex(ValueError,'diameter'):validate_config(c)

    def test_null_real_config(self):
        c=read_json(Path(__file__).resolve().parent.parent/'configs/real.example.json')
        with self.assertRaisesRegex(ValueError,'Missing required'):validate_config(c)

    def test_rank_deficient_warning(self):
        J=np.array([[1,2,0],[2,4,0],[3,6,0],[4,8,0]],float)
        r=matrix_diagnostics(J,['a','coupled_a','zero'])
        self.assertFalse(r['full_column_rank']);self.assertEqual(r['rank'],1)
        self.assertIn('zero',r['near_zero_columns']);self.assertIsNone(r['parameter_standard_deviations'])

    def test_unconfirmed_empty_wrong_ids(self):
        c=fixture_config()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'observations.csv'
            row=dict(image_id='f',view='left',point_id='0',u_px=1,v_px=1,accepted=True,review_status='needs_review')
            write_csv(path,[row])
            with self.assertRaisesRegex(ValueError,'No accepted'):observations(path,c)
            row.update(review_status='confirmed',point_id='wrong')
            write_csv(path,[row])
            with self.assertRaisesRegex(ValueError,'Unknown view or point'):observations(path,c)

    def test_insufficient_poses(self):
        c=fixture_config();state,poses=true_scene(c)
        with self.assertRaisesRegex(ValueError,'Insufficient'):calibrate(c,make_observations(c,state,poses,2))

    def test_physical_angle_gate(self):
        c=fixture_config();state,_=true_scene(c)
        result=derive_angles(state,c,False)
        self.assertIsNone(result['alpha_deg'])
        c['optimization']['mode']='physical_constrained'
        c['optimization']['release_omega']=[False]*3;c['camera']['omega_source']='measured_prior';c['camera']['alpha_sign']=1
        result=derive_angles(state,c,True)
        self.assertAlmostEqual(result['alpha_deg'],8,places=6)
        self.assertAlmostEqual(result['beta_deg'],5,places=6)

    def test_repeated_pose_data_is_weak(self):
        c=fixture_config();state,poses=true_scene(c);state['kappa']=np.zeros(2)
        repeated=np.tile(poses[0],(5,1))
        obs=make_observations(c,state,repeated,5)
        data=array_data(obs['used'],obs['frames'])
        codec=Parameters(state,obs['frames'],c);objective=Objective(codec,data)
        x=codec.pack(state,repeated);active=codec.active('B')
        J=finite_jacobian(lambda v:objective.raw(v,False).ravel(),x,active)
        r=matrix_diagnostics(J,[codec.names[i] for i in active])
        self.assertFalse(r['full_column_rank'])

    def test_correspondence_corruption_rejected(self):
        c=fixture_config();state,poses=true_scene(c)
        obs=make_observations(c,state,poses,5)
        c['validation']['frames']=[];c['optimization']['multistart']=1
        for frame in obs['frames']:
            right=[v for v in obs['used'] if v['row']['image_id']==frame and v['row']['view']=='right']
            reversed_pixels=[v['p'].copy() for v in right][::-1]
            for v,p in zip(right,reversed_pixels):v['p']=p
        with self.assertRaisesRegex(ValueError,'All initializations failed'):
            calibrate(c,obs)

    def test_noiseless_constrained_joint_solution(self):
        c=fixture_config();state,poses=true_scene(c)
        state['kappa']=np.zeros(2)
        c['camera']['K_init']=(state['K']*1.005).tolist()
        c['camera']['omega_p_deg']=np.degrees(state['omega']).tolist()
        c['optimization'].update(release_omega=[False]*3,release_radial=[False]*2,multistart=1,max_nfev=100,loss='linear',tolerance=1e-11)
        c['validation']['frames']=[]
        result=calibrate(c,make_observations(c,state,poses,5))
        self.assertLess(np.max(np.abs(result['raw_errors'])),1e-5)
        self.assertTrue(result['status']['geometry_valid'])
        np.testing.assert_allclose(result['state']['K'],state['K'],rtol=1e-5,atol=1e-5)
