import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from bbs_calib.geometry import snell,plane_projection,optical_system,scheimpflug,GeometryError,path_validity
from bbs_calib.synthetic import fixture_config,true_scene
from bbs_calib.projection import project,object_points
from bbs_calib.reconstruction import reconstruct_points,undistort
from bbs_calib.initialization import initialize_pose
from bbs_calib.config import board_points


def independent_trace(X,prism,omega,branch):
    """Independent segment intersections, no B matrices or production Snell call."""
    R=Rotation.from_rotvec(omega).as_matrix()
    incoming=R.T@np.array([0.,0.,1.])
    # Back interface is horizontal; Snell tangential x/y components scale by 1/n.
    d2=np.r_[incoming[:2]/prism['refractive_index'],0.]
    d2[2]=np.sqrt(1-np.dot(d2,d2))
    tau=(-1 if branch=='L' else 1)*np.tan(np.radians(prism['wedge_'+branch+'_deg']))
    N=np.array([tau,0.,1.]);N/=np.linalg.norm(N)
    theta=np.arccos(np.dot(d2,N))
    tangent=d2-np.dot(d2,N)*N
    if np.linalg.norm(tangent)<1e-14:
        d1=N
    else:
        tangent/=np.linalg.norm(tangent)
        theta_out=np.arcsin(prism['refractive_index']*np.sin(theta))
        d1=np.cos(theta_out)*N+np.sin(theta_out)*tangent
    front=X+((prism['thickness_mm']-tau*X[0]-X[2])/(tau*d1[0]+d1[2]))*d1
    back=front-front[2]/d2[2]*d2
    return (R@back)[:2]


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.c=fixture_config();self.state,self.poses=true_scene(self.c)
        self.cal=dict(state=self.state,view_branches={v:self.c['views'][v]['branch'] for v in ('left','right')},radius_limit_mm=10.)

    def test_normal_snell(self):
        np.testing.assert_allclose(snell([0,0,1],[0,0,1],1,1.5),[0,0,1])

    def test_reciprocity(self):
        d=np.array([.25,.15,np.sqrt(1-.25**2-.15**2)])
        out=snell(d,[0,0,1],1,1.5)
        np.testing.assert_allclose(snell(-out,[0,0,-1],1.5,1),-d,atol=1e-14)

    def test_tir_and_normal_orientation(self):
        with self.assertRaisesRegex(GeometryError,'Total internal'):
            snell([.9,0,np.sqrt(.19)],[0,0,1],1.5,1)
        with self.assertRaises(GeometryError):snell([0,0,-1],[0,0,1],1,1.5)

    def test_plane_intersection_and_parallel(self):
        P=plane_projection([.1,.2,1],[.2,0,1],2.)
        v=P@np.array([3.,4.,8.,1.])
        self.assertAlmostEqual(.2*v[0]+v[2],2.,places=12)
        with self.assertRaises(GeometryError):plane_projection([1,0,0],[0,0,1],0)

    def test_independent_ray_matrices(self):
        rng=np.random.default_rng(2026)
        for _ in range(12):
            omega=rng.uniform(-.04,.04,3)
            system=optical_system(self.state['prism'],omega)
            X=rng.uniform([-.5,-.5,15],[.5,.5,22])
            for branch in ('L','R'):
                ref=independent_trace(X,self.state['prism'],omega,branch)
                np.testing.assert_allclose(system[branch]['G']@np.r_[X,1],ref,atol=2e-12)
                self.assertAlmostEqual((system[branch]['B']@np.r_[X,1])[2],0.,places=12)

    def test_zero_scheimpflug(self):np.testing.assert_allclose(scheimpflug(0,0),np.eye(2))

    def test_depth_degeneracy(self):
        self.state['prism']=dict(self.state['prism'],wedge_L_deg=0,wedge_R_deg=0)
        r=reconstruct_points([[1,2]],[[1,2]],self.cal)
        self.assertFalse(r['geometry_valid']);self.assertIsNone(r['X_mm'])

    def test_projection_reconstruction_and_xyz_translation(self):
        X=np.array([[-.2,.1,20],[.3,-.2,21]])
        def rec(X):
            pixels=[project(X,self.cal['view_branches'][v],self.state) for v in ('left','right')]
            return reconstruct_points(*pixels,self.cal)
        r=rec(X);self.assertTrue(r['geometry_valid'])
        np.testing.assert_allclose(r['X_mm'],X,atol=2e-11)
        for delta in np.eye(3)*.04:
            np.testing.assert_allclose(rec(X+delta)['X_mm']-r['X_mm'],np.tile(delta,(2,1)),atol=2e-11)

    def test_single_view_refused(self):
        with self.assertRaises(ValueError):reconstruct_points(None,[[1,2]],self.cal)

    def test_omega_updates_directions_and_derivative(self):
        p=self.state['prism'];omega=self.state['omega']
        a=optical_system(p,omega);b=optical_system(p,omega+[.003,0,0])
        self.assertGreater(np.linalg.norm(a['d2']-b['d2']),1e-5)
        self.assertGreater(np.linalg.norm(a['R']['d1']-b['R']['d1']),1e-5)
        X=np.array([.1,.2,20.]);h=1e-5
        for axis in np.eye(3):
            numerical=(optical_system(p,omega+h*axis)['R']['G']-optical_system(p,omega-h*axis)['R']['G'])@np.r_[X,1]/(2*h)
            ref=(independent_trace(X,p,omega+h*axis,'R')-independent_trace(X,p,omega-h*axis,'R'))/(2*h)
            np.testing.assert_allclose(numerical,ref,rtol=1e-6,atol=1e-7)

    def test_distortion_nonmonotone_refused(self):
        with self.assertRaisesRegex(GeometryError,'not monotone'):
            undistort([[.2,0]],[-1,0],1,2)

    def test_pose_initialization_center_subtraction(self):
        ids,P=board_points(self.c);pose=self.poses[1]
        X=object_points(P,pose)
        state=dict(self.state,kappa=np.zeros(2))
        allP=np.vstack([P,P]);pixels=np.vstack([project(X,s,state) for s in ('L','R')])
        estimate,_=initialize_pose(allP,pixels,['L']*len(P)+['R']*len(P),state)
        np.testing.assert_allclose(object_points(P,estimate),X,atol=1e-10)

    def test_nonplanar_pose(self):
        _,P=board_points(self.c);P=P.copy();P[:,2]=np.random.default_rng(34).normal(0,.1,len(P))
        pose=self.poses[2];X=object_points(P,pose);state=dict(self.state,kappa=np.zeros(2))
        est,_=initialize_pose(np.vstack([P,P]),np.vstack([project(X,s,state) for s in ('L','R')]),['L']*len(P)+['R']*len(P),state)
        np.testing.assert_allclose(object_points(P,est),X,atol=1e-10)

    def test_validity_wrong_half(self):
        system=optical_system(self.state['prism'],self.state['omega'])
        valid,_,_=path_validity(np.array([[100,0,20]]),'L',system,self.state['prism'])
        self.assertFalse(valid[0])
