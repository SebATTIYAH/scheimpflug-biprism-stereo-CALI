import unittest
import tempfile
from copy import deepcopy
from pathlib import Path
import numpy as np
from bbs_calib.image_inputs import discover_frames,origin_for,crop_origins
from bbs_calib.io import write_csv,read_csv,save_image,new_output
from bbs_calib.config import validate_config,board_points
from bbs_calib.synthetic import fixture_config,true_scene,render_frame
from bbs_calib.detection import detect_dataset
from bbs_calib.optimization import observations
from bbs_calib.geometry import optical_system
from bbs_calib.projection import project
from bbs_calib.reconstruction import reconstruct_points


class CroppedTests(unittest.TestCase):
    def fixture(self,d):
        c=fixture_config();c['images'].update(mode='paired_crops',directory=str(d),pattern='*.png')
        c['review'].update(anchors_csv=None,anchor_coordinates='crop_local')
        for v,offset in [('left',[0,0]),('right',[500,0])]:
            c['views'][v].update(roi=None,crop_origin_px=offset)
        return c

    def test_suffix_pairing_missing_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.fixture(root)
            for name in ('test_01_0.png','test_01_1.png','test_02_0.png'):
                save_image(root/name,np.zeros((8,10),np.uint8))
            with self.assertRaisesRegex(ValueError,'Missing _0/_1'):discover_frames(c)
            save_image(root/'test_02_1.png',np.zeros((8,10),np.uint8))
            frames=discover_frames(c)
            self.assertEqual([f['image_id'] for f in frames],['test_01','test_02'])
            self.assertTrue(frames[0]['left'].stem.endswith('_0'))
            save_image(root/'test_01_0.tif',np.zeros((8,10),np.uint8))
            c['images']['pattern']='*.*'
            with self.assertRaisesRegex(ValueError,'Duplicate crop'):discover_frames(c)

    def test_per_frame_origins(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.fixture(d);path=Path(d)/'origins.csv'
            write_csv(path,[dict(image_id='f',view='left',x0_px=12,y0_px=20)])
            c['images']['crop_origins_csv']=str(path)
            np.testing.assert_allclose(origin_for(c,'f','left'),[12,20])
            np.testing.assert_allclose(origin_for(c,'g','right'),[500,0])

    def test_matlab_midpoint_even_odd_and_conflicts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.fixture(root)
            c['images']['crop_layout']='matlab_midpoint_vertical'
            for v in ('left','right'):c['views'][v]['crop_origin_px']=None
            for name,wr in [('even_bmp',10),('odd_tif',11)]:
                save_image(root/(name+'_0.png'),np.zeros((8,10),np.uint16))
                save_image(root/(name+'_1.png'),np.zeros((8,wr),np.uint16))
            table=crop_origins(c)
            self.assertEqual(table['odd_tif'],dict(left=[0,0],right=[10,0]))
            self.assertEqual(table['even_bmp'],table['odd_tif'])
            c['views']['right']['crop_origin_px']=[11,0]
            with self.assertRaisesRegex(ValueError,'conflicts'):crop_origins(c)

    def test_matlab_midpoint_invalid_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.fixture(root)
            c['images']['crop_layout']='matlab_midpoint_vertical'
            for v in ('left','right'):c['views'][v]['crop_origin_px']=None
            save_image(root/'f_0.png',np.zeros((8,10),np.uint8))
            for shape in [(9,10),(8,9),(8,12)]:
                save_image(root/'f_1.png',np.zeros(shape,np.uint8))
                with self.assertRaisesRegex(ValueError,'dimensions'):crop_origins(c)

    def test_cropped_detection_matches_full_frame(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.fixture(root)
            state,poses=true_scene(c)
            image,truth=render_frame(c,state,poses[0],np.random.default_rng(7))
            save_image(root/'sample_0.png',image[:,:500]);save_image(root/'sample_1.png',image[:,500:])
            anchors=[]
            for r in truth:
                if r['point_id'] in ('0','5','24'):
                    anchors.append(dict(image_id='sample',view=r['view'],point_id=r['point_id'],
                                        u_px=r['u_px']-(500 if r['view']=='right' else 0),v_px=r['v_px']))
            write_csv(root/'anchors.csv',anchors);c['review']['anchors_csv']=str(root/'anchors.csv')
            out=new_output(root/'detected');det=detect_dataset(c,out)
            self.assertEqual(det['confirmed'],60)
            obs=observations(out/'observations.csv',c)
            self.assertEqual(obs['frames'],['sample'])
            for row in read_csv(out/'observations.csv'):
                if row['accepted']=='True':
                    self.assertEqual(row['coordinate_space'],'full_frame')
                    offset=500 if row['view']=='right' else 0
                    self.assertAlmostEqual(float(row['u_px'])-float(row['source_u_px']),offset)
            # No invented zero offsets: local detection is allowed, calibration is blocked.
            c['ridge_masks']=[]
            for v in ('left','right'):c['views'][v]['crop_origin_px']=None
            unknown=new_output(root/'unknown_offsets');det=detect_dataset(c,unknown)
            self.assertEqual(det['confirmed'],60)
            with self.assertRaisesRegex(ValueError,'Crop origin missing'):
                observations(unknown/'observations.csv',c)
            # Supplying offsets later restores saved local observations without redetection.
            c['views']['left']['crop_origin_px']=[0,0];c['views']['right']['crop_origin_px']=[500,0]
            restored=observations(unknown/'observations.csv',c)
            np.testing.assert_allclose([v['p'] for v in restored['used']],[v['p'] for v in obs['used']])

    def test_crop_local_reconstruction(self):
        c=fixture_config();state,_=true_scene(c)
        cal=dict(state=state,input_mode='paired_crops',default_crop_origins=dict(left=[15,30],right=[480,20]),
                 view_branches=dict(left='R',right='L'),radius_limit_mm=10.)
        X=np.array([[.1,.2,20.],[-.2,-.1,21.]])
        a=project(X,'R',state)-[15,30];b=project(X,'L',state)-[480,20]
        result=reconstruct_points(a,b,cal,pixel_coordinates='crop_local')
        np.testing.assert_allclose(result['X_mm'],X,atol=1e-10)

    def test_thickness_only_affine_offset(self):
        c=fixture_config();state,_=true_scene(c)
        p1=state['prism'];p2=dict(p1,thickness_mm=p1['thickness_mm']+.8)
        for omega in (np.zeros(3),state['omega']):
            a=optical_system(p1,omega);b=optical_system(p2,omega)
            for s in ('L','R'):
                np.testing.assert_allclose(a[s]['M'],b[s]['M'],atol=1e-14)
                np.testing.assert_allclose(a[s]['d1'],b[s]['d1'],atol=1e-14)
                self.assertGreater(np.linalg.norm(a[s]['b']-b[s]['b']),1e-5)

    def test_upright_thickness_depth_origin_equivalence(self):
        c=fixture_config();state,_=true_scene(c);state['omega']=np.zeros(3)
        new=deepcopy(state);new['prism']['thickness_mm']+=.8
        X=np.array([[.1,.2,20.],[-.2,-.1,21.]])
        for s in ('L','R'):
            np.testing.assert_allclose(project(X,s,state),project(X+[0,0,.8],s,new),atol=1e-11)
        # An initial zero is not a fixed zero: generic tilted geometry changes the result.
        state['omega']=np.array([.04,.03,.02]);new['omega']=state['omega']
        self.assertGreater(np.linalg.norm(project(X,'L',state)-project(X+[0,0,.8],'L',new)),.01)
