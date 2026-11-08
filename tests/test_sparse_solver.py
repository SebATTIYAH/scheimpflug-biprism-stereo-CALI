import contextlib,io
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from bbs_calib.sparse_solver import GroupedJacobian,block_rank


class SparseSolverTests(unittest.TestCase):
    def test_grouped_threads_match_derivatives(self):
        A=np.array([[1,2,3,0,0],[2,3,1,0,0],[1,0,0,4,2],[3,0,0,1,4]],float)
        active=np.array([0,8,9,14,15]);x=np.arange(5)*.2
        expected=A*np.cos(x)
        for workers in (1,4):
            j=GroupedJacobian(lambda x:A@np.sin(x),csr_matrix(A!=0),active,workers)
            try:np.testing.assert_allclose(j(x).toarray(),expected,atol=1e-9)
            finally:j.close()

    def test_bounds_one_sided_and_overlap_rejected(self):
        x=np.array([0.,1.]);A=csr_matrix(np.eye(2))
        j=GroupedJacobian(lambda x:x*x,A,np.array([8,14]),low=[0,0],high=[1,1])
        try:np.testing.assert_allclose(j(x).toarray(),np.diag(2*x),atol=1e-5)
        finally:j.close()
        with self.assertRaisesRegex(ValueError,'Overlapping'):GroupedJacobian(lambda x:x,csr_matrix(np.ones((2,2))),np.array([8,14]))

    def test_block_rank_and_global_projection(self):
        A=np.array([[1,1,0],[2,1,0],[1,0,2],[4,0,1]],float)
        active=np.array([0,8,14]);frames=np.array([0,1])
        r,g=block_rank(csr_matrix(A),active,frames,['global','p0','p1'])
        self.assertEqual(r['rank'],np.linalg.matrix_rank(A));self.assertTrue(g['full_column_rank'])
        projected=A[:,:1]-A[:,1:]@np.linalg.lstsq(A[:,1:],A[:,:1],rcond=None)[0]
        np.testing.assert_allclose(g['singular_values'],np.linalg.svd(projected,compute_uv=False),atol=1e-12)
        A[:,0]=A[:,1]+A[:,2];r,g=block_rank(csr_matrix(A),active,frames,['g','p0','p1'])
        self.assertFalse(g['full_column_rank']);self.assertEqual(r['rank'],2)
        # Near-zero reduced global columns need an absolute scale-aware cutoff.
        self.assertLess(np.linalg.norm(A[:,:1]-A[:,1:]@np.linalg.lstsq(A[:,1:],A[:,:1],rcond=None)[0]),1e-12)

    def fixture(self):
        from bbs_calib.synthetic import fixture_config,true_scene
        from bbs_calib.optimization import Parameters,Objective,array_data
        from test_calibration import make_observations
        c=fixture_config();state,poses=true_scene(c);obs=make_observations(c,state,poses,5)
        codec=Parameters(state,obs['frames'],c);obj=Objective(codec,array_data(obs['used'],obs['frames']))
        return c,state,poses,obs,codec,obj

    def test_vector_projection_matches_legacy_and_jacobian(self):
        from bbs_calib.optimization import finite_jacobian
        c,state,poses,obs,codec,obj=self.fixture();x=codec.pack(state,np.array(poses[:5]))
        old=obj.raw(x);c['optimization']['resolved_backend']='sparse'
        np.testing.assert_allclose(obj.raw(x),old,atol=1e-11)
        active=codec.active('C');j=GroupedJacobian(obj,obj.sparsity(active),active,workers=2)
        try:actual=j(x).toarray()
        finally:j.close()
        expected=finite_jacobian(obj,x,active)
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=1e-5)

    def test_sparse_synthetic_recovery(self):
        from bbs_calib.optimization import calibrate
        from test_calibration import make_observations
        c,state,poses,obs,codec,obj=self.fixture();state['kappa']=np.zeros(2)
        c['camera']['K_init']=(state['K']*1.005).tolist();c['camera']['omega_p_deg']=np.degrees(state['omega']).tolist()
        c['optimization'].update(backend='sparse',workers=2,release_omega=[False]*3,release_radial=[False]*2,
            solver_profile='enhanced',multistart=1,max_nfev=100,refine_nfev=30,loss='linear',tolerance=1e-11)
        c['validation']['frames']=[]
        with contextlib.redirect_stdout(io.StringIO()):result=calibrate(c,make_observations(c,state,poses,5))
        np.testing.assert_allclose(result['state']['K'],state['K'],atol=2e-5)
        self.assertEqual(result['diagnostics']['execution']['backend'],'sparse')
        self.assertTrue(result['status']['geometry_valid'])

    def test_cli_forwarding(self):
        from bbs_calib.step_cli import main
        with patch('bbs_calib.step_cli.resolve_mode',return_value='prism'),patch('bbs_calib.workflow_cli.main',return_value=0) as run,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['fit','--source','in','--output','out','--backend','sparse','--workers','4']),0)
            self.assertEqual(run.call_args.args[0][-4:],['--backend','sparse','--workers','4'])
