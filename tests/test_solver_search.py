import contextlib
import io
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from scipy.optimize import least_squares
from bbs_calib.solver_search import robust_cost,refine,candidate_key,stability,options


class SolverSearchTests(unittest.TestCase):
    def test_cost_matches_scipy_for_all_losses(self):
        r=np.array([-.001,.2,-.8,2.,10.])
        for loss in ('linear','soft_l1','huber','cauchy','arctan'):
            result=least_squares(lambda x:r.copy(),np.array([0.]),loss=loss,f_scale=.7)
            self.assertAlmostEqual(robust_cost(r,loss,.7),result.cost,places=12)

    def test_robust_ranking_differs_from_unrobust(self):
        a=np.array([10.,0.]);b=np.array([6.,6.])
        self.assertGreater(robust_cost(a),robust_cost(b))
        self.assertLess(robust_cost(a,'soft_l1'),robust_cost(b,'soft_l1'))

    def test_refinement_improves_and_respects_bounds(self):
        x,r=refine(lambda x:x-2,np.array([0.]),np.array([-1.]),np.array([3.]))
        self.assertTrue(r['accepted']);self.assertLess(r['after_cost'],r['before_cost'])
        np.testing.assert_allclose(x,[2.],atol=1e-8)

    def test_reject_invalid_unconverged_or_worse_refinement(self):
        initial=np.array([1.]);low=np.array([-10.]);high=np.array([10.])
        for success,value,valid in [(True,0.,False),(False,0.,True),(True,5.,True)]:
            result=SimpleNamespace(x=np.array([value]),success=success,optimality=1.,nfev=1,message='test',status=1)
            with patch('bbs_calib.solver_search.least_squares',return_value=result):
                x,r=refine(lambda x:x,initial,low,high,valid=lambda x:valid)
            self.assertFalse(r['accepted']);np.testing.assert_array_equal(x,initial)

    def test_geometry_and_convergence_precede_score(self):
        valid=dict(geometry_valid=True,converged=True,score=100.)
        self.assertLess(candidate_key(valid),candidate_key(dict(geometry_valid=False,converged=True,score=0.)))
        self.assertLess(candidate_key(valid),candidate_key(dict(geometry_valid=True,converged=False,score=0.)))

    def test_stability_warns_near_equal_cost_different_angles(self):
        records=[dict(trial=i,geometry_valid=True,converged=True,score=10+i*.001,values=dict(total_tilt_deg=angle)) for i,angle in enumerate([2.,5.])]
        r=stability(records,0,12)
        self.assertEqual(r['status'],'near_equal_cost_different_angles')
        self.assertEqual(r['near_optimal_ranges']['total_tilt_deg']['spread'],3.)
        self.assertFalse(r['confidence_interval']);self.assertFalse(r['global_uniqueness_proven'])
        self.assertEqual(stability(records[:1],0,12)['status'],'insufficient_converged_near_optimal_starts')

    def test_invalid_search_options(self):
        for kw in [dict(starts=0),dict(refine_nfev=0),dict(seed=-1),dict(profile='baseline',starts=2)]:
            with self.assertRaises(ValueError):options(**kw)

    def test_prism_search_reproducible_and_fixed_parameters_preserved(self):
        from bbs_calib.synthetic import fixture_config,true_scene
        from bbs_calib.optimization import calibrate
        from test_calibration import make_observations
        c=fixture_config();state,poses=true_scene(c);state['kappa']=np.zeros(2)
        c['camera']['K_init']=(state['K']*1.005).tolist();c['camera']['omega_p_deg']=np.degrees(state['omega']).tolist()
        c['optimization'].update(release_omega=[False]*3,release_radial=[False]*2,multistart=2,solver_profile='enhanced',
            search_seed=23,refine_nfev=30,max_nfev=100,loss='linear',tolerance=1e-11)
        c['validation']['frames']=[];obs=make_observations(c,state,poses,5)
        with contextlib.redirect_stdout(io.StringIO()):a=calibrate(c,obs);b=calibrate(c,obs)
        self.assertEqual(a['training_frames'],b['training_frames'])
        np.testing.assert_allclose(a['state']['K'],state['K'],atol=1e-5)
        np.testing.assert_allclose(a['state']['K'],b['state']['K'],atol=1e-10)
        np.testing.assert_array_equal(a['state']['omega'],state['omega'])
        np.testing.assert_array_equal(a['state']['kappa'],[0.,0.])
        report=a['diagnostics']['optimizer_search'];self.assertEqual(report['attempted'],2)
        self.assertEqual(report['eligible_count'],2)
        self.assertEqual(report['selected_trial'],min(report['candidates'],key=candidate_key)['trial'])
        self.assertNotEqual(report['candidates'][0]['initial_K'],report['candidates'][1]['initial_K'])

    def test_single_search_recovers_synthetic_geometry(self):
        from test_single_camera import SingleCameraTests
        from bbs_calib.single_camera import calibrate_single_search
        K,c,frames=SingleCameraTests().frames()
        with contextlib.redirect_stdout(io.StringIO()):
            result=calibrate_single_search(frames,[1600,1200],options(starts=2,refine_nfev=30),release_center=False)
        np.testing.assert_allclose(result['K'],K,atol=1e-5)
        np.testing.assert_array_equal(result['c'],c)
        self.assertEqual(result['optimizer_search']['attempted'],2)
        self.assertEqual(result['optimizer_search']['eligible_count'],2)

    def test_step_cli_forwards_solver_controls(self):
        from bbs_calib.step_cli import main
        with patch('bbs_calib.step_cli.resolve_mode',return_value='prism'),patch('bbs_calib.workflow_cli.main',return_value=0) as driver,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['fit','--source','in','--output','out','--solver','enhanced','--starts','7','--search-seed','31','--refine-nfev','80']),0)
            args=driver.call_args.args[0]
            self.assertEqual(args[-8:],['--solver','enhanced','--starts','7','--search-seed','31','--refine-nfev','80'])
