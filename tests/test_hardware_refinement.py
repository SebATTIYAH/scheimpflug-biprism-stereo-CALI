import contextlib
import io
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from bbs_calib.config import validate_config, board_points
from bbs_calib.hardware_refinement import apply_mode, specifications, report
from bbs_calib.optimization import Parameters, Objective, array_data, finite_jacobian, calibrate
from bbs_calib.synthetic import fixture_config, true_scene
from bbs_calib.sparse_solver import GroupedJacobian, diagnostics, block_rank
from bbs_calib.projection import project, object_points
from test_calibration import make_observations


class HardwareRefinementTests(unittest.TestCase):
    def fixture(self, mode='all'):
        c=fixture_config();apply_mode(c,mode)
        state,poses=true_scene(c)
        obs=make_observations(c,state,poses,5)
        codec=Parameters(state,obs['frames'],c)
        obj=Objective(codec,array_data(obs['used'],obs['frames']))
        return c,state,poses[:5],codec,obj

    def test_defaults_modes_and_validation(self):
        c=fixture_config();self.assertEqual(specifications(c),[])
        apply_mode(c,'wedges');self.assertEqual(len(specifications(c)),2)
        apply_mode(c,'all');self.assertEqual(len(specifications(c)),3);validate_config(c)
        saved=deepcopy(c);apply_mode(c,None);self.assertEqual(c,saved)
        for value in (0,-1,True,float('nan'),float('inf'),'0.1'):
            bad=deepcopy(c);bad['optimization']['hardware_refinement']['parameters']['wedge_L_deg']['max_delta']=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate_config(bad)
        for cfg in (None,{'enabled':'yes'}, {'enabled':True,'parameters':{}},
                    {'enabled':True,'parameters':{'thickness_mm':{'max_delta':1,'prior_sigma':1}}}):
            bad=deepcopy(c);bad['optimization']['hardware_refinement']=cfg
            with self.assertRaises(ValueError):validate_config(bad)
        c['optimization']['hardware_refinement']['parameters']['wedge_L_deg']['max_delta']=80
        with self.assertRaises(ValueError):validate_config(c)
        apply_mode(c,'off');validate_config(c);self.assertEqual(specifications(c),[])

    def test_layout_bounds_and_stage_release(self):
        c,state,poses,codec,obj=self.fixture();original=deepcopy(c)
        self.assertEqual(codec.n_global,11)
        x=codec.pack(state,poses);x[8:11]=[.2,-.3,.4]
        fitted,back=codec.unpack(x)
        np.testing.assert_allclose(back,poses)
        np.testing.assert_allclose(codec.pack(fitted,back),x)
        self.assertEqual(c,original)
        lo,hi=codec.bounds(x)
        np.testing.assert_allclose(lo[8:11],-1);np.testing.assert_allclose(hi[8:11],1)
        self.assertEqual(lo[16],0)
        for stage in ('A','B'):
            self.assertFalse(set(range(8,11)) & set(codec.active(stage)))
        self.assertTrue(set(range(8,11)) <= set(codec.active('C')))
        np.testing.assert_allclose(obj(x)[-3:],[.4,-.6,.8])

    def test_vector_grouped_and_dense_jacobian(self):
        c,state,poses,codec,obj=self.fixture();x=codec.pack(state,poses);x[8:11]=[.1,-.2,.3]
        reference=obj.raw(x);c['optimization']['resolved_backend']='sparse'
        np.testing.assert_allclose(obj.raw(x),reference,atol=1e-11)
        active=codec.active('C');frozen=x.copy()
        def fun(y):
            full=frozen.copy();full[active]=y;return obj(full)
        expected=finite_jacobian(obj,x,active)
        for workers in (1,4):
            jac=GroupedJacobian(fun,obj.sparsity(active),active,workers,n_global=codec.n_global)
            try:actual=jac(x[active]).toarray()
            finally:jac.close()
            np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=2e-5)
        raw,aug,glob=diagnostics(obj,x,active)
        self.assertEqual(raw['columns'],len(active))
        self.assertEqual(glob['columns'],sum(active<codec.n_global))
        self.assertEqual(aug['columns'],raw['columns'])

    def test_generalized_block_rank(self):
        from scipy.sparse import csr_matrix
        A=np.array([[1,1,0],[2,1,0],[1,0,2],[4,0,1]],float)
        raw,glob=block_rank(csr_matrix(A),np.array([8,11,17]),np.array([0,1]),['n','p0','p1'],n_global=11)
        self.assertEqual(raw['rank'],3);self.assertEqual(glob['columns'],1)

    def test_cli_forwarding_and_single_rejection(self):
        from bbs_calib.step_cli import main
        with patch('bbs_calib.step_cli.resolve_mode',return_value='prism'),patch('bbs_calib.workflow_cli.main',return_value=0) as run,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['fit','--source','in','--output','out','--refine-hardware','wedges']),0)
            self.assertEqual(run.call_args.args[0][-2:],['--refine-hardware','wedges'])
        with patch('bbs_calib.step_cli.resolve_mode',return_value='single'),patch('bbs_calib.step_cli.choose_directory') as dialog,contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['fit','--source','in','--refine-hardware','all']),2)
            dialog.assert_not_called()

    def test_report_and_reconstruction_use_fitted_hardware(self):
        from bbs_calib.reporting import calibration_document, save_results
        from bbs_calib.reconstruction import reconstruct_points
        from bbs_calib.io import write_json,read_json
        from bbs_calib.workflow_cli import inspect_result
        c,state,poses,codec,obj=self.fixture()
        state['prism']['wedge_L_deg']+=.04
        hardware=report(c,state,False)
        result=dict(state=state,poses={},diagnostics={'hardware_refinement':hardware},
                    status={'identifiability_passed':False,'geometry_valid':True},
                    mechanical_angles={},training_frames=[],validation_frames=[])
        cal=calibration_document(result,c)
        self.assertEqual(cal['parameter_status']['prism.wedge_L_deg']['status'],'estimated_with_nominal_prior')
        self.assertEqual(cal['parameter_status']['prism.thickness_mm']['status'],'fixed')
        self.assertEqual(cal['config_snapshot']['prism']['wedge_L_deg'],c['prism']['wedge_L_deg'])
        X=object_points(board_points(c)[1],poses[0])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'calibration.json';write_json(path,cal);saved=read_json(path)
            answer=reconstruct_points(project(X,c['views']['left']['branch'],state),project(X,c['views']['right']['branch'],state),saved)
            np.testing.assert_allclose(answer['X_mm'],X,atol=1e-9)
            self.assertFalse(inspect_result(path)['hardware_refinement']['independently_measured'])

    def test_synthetic_single_wedge_recovery_with_known_other_globals(self):
        # Controlled test of implementation, NOT proof that all hardware is jointly identifiable.
        c=fixture_config();apply_mode(c,'wedges')
        c['optimization']['hardware_refinement']['parameters']={'wedge_L_deg':{'max_delta':.2,'prior_sigma':10.}}
        state,poses=true_scene(c);state['kappa']=np.zeros(2);state['prism']['wedge_L_deg']+=.06
        c['camera']['K_init']=state['K'].tolist();c['camera']['omega_p_deg']=np.degrees(state['omega']).tolist()
        c['optimization'].update(backend='sparse',workers=2,release_K=[False]*3,release_omega=[False]*3,
            release_radial=[False]*2,multistart=1,max_nfev=150,loss='linear',tolerance=1e-10)
        c['validation']['frames']=[]
        for backend,workers in (('sparse',2),('legacy',1)):
            c['optimization'].update(backend=backend,workers=workers)
            with self.subTest(backend=backend),contextlib.redirect_stdout(io.StringIO()):
                result=calibrate(c,make_observations(c,state,poses,5))
                self.assertLess(abs(result['state']['prism']['wedge_L_deg']-state['prism']['wedge_L_deg']),.001)
                self.assertLess(np.sqrt(np.mean(result['raw_errors']**2)),.001)
                self.assertIn('hardware_refinement',result['diagnostics'])
                self.assertTrue(result['status']['geometry_valid'])
                self.assertTrue(result['status']['optimization_converged'])
                self.assertIsNone(result['mechanical_angles']['alpha_deg'])

    def test_joint_all_hardware_fit_save_and_inspect(self):
        from bbs_calib.reporting import save_results
        from bbs_calib.io import read_json
        from bbs_calib.workflow_cli import inspect_result
        from bbs_calib.reconstruction import reconstruct_points
        c=fixture_config();apply_mode(c,'all')
        state,poses=true_scene(c);state['kappa']=np.zeros(2)
        state['prism']['wedge_L_deg']+=.035;state['prism']['wedge_R_deg']-=.025
        state['prism']['refractive_index']+=.0005
        c['camera']['K_init']=state['K'].tolist();c['camera']['omega_p_deg']=np.degrees(state['omega']).tolist()
        c['optimization'].update(backend='sparse',workers=2,release_radial=[False]*2,
            solver_profile='enhanced',multistart=1,max_nfev=100,refine_nfev=40)
        c['validation']['frames']=['frame_04']
        obs=make_observations(c,state,poses,5)
        with contextlib.redirect_stdout(io.StringIO()):result=calibrate(c,obs)
        self.assertTrue(result['status']['geometry_valid'])
        self.assertFalse(result['diagnostics']['hardware_refinement']['independently_measured'])
        self.assertIn('prism.wedge_L_deg',result['diagnostics']['optimizer_search']['candidates'][0]['values'])
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);save_results(result,c,obs,out,'test')
            saved=read_json(out/'calibration.json')
            report_file=read_json(out/'hardware_refinement.json')
            self.assertEqual(inspect_result(out)['hardware_refinement'],report_file)
            for spec in specifications(c):
                fitted=saved['state']['prism'][spec['name']]
                self.assertGreaterEqual(fitted,spec['lower']);self.assertLessEqual(fitted,spec['upper'])
                self.assertEqual(saved['prism'][spec['name']],fitted)
            fitted=result['state'];X=object_points(board_points(c)[1],list(result['poses'].values())[0])
            reconstructed=reconstruct_points(project(X,c['views']['left']['branch'],fitted),
                project(X,c['views']['right']['branch'],fitted),saved)
            np.testing.assert_allclose(reconstructed['X_mm'],X,atol=1e-8)

    def test_cli_validates_before_creating_output(self):
        from bbs_calib.workflow_cli import main
        from bbs_calib.io import write_json
        c=fixture_config();c['prism']['wedge_L_deg']=.1
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder);write_json(source/'config.json',c);(source/'observations.csv').touch()
            out=source/'never_created'
            with contextlib.redirect_stderr(io.StringIO()):
                code=main(['prism','fit','--source',str(source),'--output',str(out),
                           '--confirm-order','--refine-hardware','wedges'])
            self.assertEqual(code,2);self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
