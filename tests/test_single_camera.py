import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from bbs_calib.geometry import scheimpflug
from bbs_calib.single_camera import intrinsic_init,projection,pose_init,calibrate_single

class SingleCameraTests(unittest.TestCase):
    def frames(self):
        K=-30*scheimpflug(*np.radians([7,4]));c=np.array([799.5,599.5])
        P=np.array([[j*3,i*3,0] for i in range(7) for j in range(6)],float);frames=[]
        for i,angles in enumerate([[-20,-15,5],[15,-20,-8],[-10,25,12],[25,12,-15],[-25,8,20],[8,-30,-20],[28,-22,3],[-18,28,-5]]):
            R=Rotation.from_euler('xyz',angles,degrees=True);pose=np.r_[R.as_rotvec(),-R.apply(P.mean(axis=0))[:2]]
            frames.append(dict(id=str(i),P=P,p=projection(P,pose,K,c,np.zeros(3),60),ids=list(range(len(P)))))
        return K,c,frames
    def test_hu_equation20_recovers_intrinsics(self):
        K,c,frames=self.frames();estimate,d=intrinsic_init(frames)
        np.testing.assert_allclose(estimate,K,atol=1e-7);self.assertEqual(d['design_rank'],4)
        for f in frames:
            pose=pose_init(f,K,c)
            np.testing.assert_allclose(projection(f['P'],pose,K,c,np.zeros(3),60),f['p'],atol=1e-8)
    def test_single_camera_nonlinear_without_biprism(self):
        K,c,frames=self.frames();r=calibrate_single(frames,[1600,1200],release_center=False,diagnostics=False)
        np.testing.assert_allclose(r['K'],K,atol=1e-6)
        self.assertLess(r['stages'][-1]['rms_px'],1e-6)
        self.assertFalse(r['tz_observable'])
    def test_degenerate_poses_rejected(self):
        K,c,frames=self.frames()
        with self.assertRaisesRegex(ValueError,'rank'):intrinsic_init([frames[0]]*8)

    def test_joint_radial_and_center_recovery(self):
        K,c,frames=self.frames();c=c+[-22,17];d=np.array([.006,-.003,.001])
        for i,f in enumerate(frames):
            pose=pose_init(f,K,np.array([799.5,599.5]));pose[3:]+=np.array([18*np.sin(i),15*np.cos(i)])
            f['p']=projection(f['P'],pose,K,c,d,60)
        r=calibrate_single(frames,[1600,1200],release_center=True,diagnostics=False)
        self.assertTrue(r['stages'][-1]['success'])
        np.testing.assert_allclose(r['K'],K,atol=1e-5)
        np.testing.assert_allclose(r['c'],c,atol=.01)
        self.assertLess(r['stages'][-1]['rms_px'],1e-5)
