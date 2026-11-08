import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from bbs_calib.io import read_json,write_json,save_image
from bbs_calib.review_images import resize_points,draw_center,regenerate
from bbs_calib.step_cli import main


class ReviewImageTests(unittest.TestCase):
    def test_pixel_center_resize(self):
        p=np.array([[0.,0.],[531.9485,1005.6197]])
        scale=np.array([375/1688,600/2704])
        np.testing.assert_allclose(resize_points(p,(2704,1688),(600,375)),(p+.5)*scale-.5)
        np.testing.assert_allclose(resize_points(p,(2704,1688),(2704,1688)),p)
        np.testing.assert_allclose(resize_points(resize_points(p,(2704,1688),(600,375)),(600,375),(2704,1688)),p,atol=1e-12)

    def test_draw_center_does_not_round_to_integer(self):
        a=np.zeros((30,30,3),np.uint8);b=a.copy()
        draw_center(a,[12.1,13.1]);draw_center(b,[12.4,13.4])
        self.assertFalse(np.array_equal(a,b))

    def test_resize_matches_sampled_image_centroid(self):
        yy,xx=np.mgrid[:160,:200];center=np.array([83.37,74.21])
        dot=np.exp(-((xx-center[0])**2+(yy-center[1])**2)/(2*6**2)).astype(np.float32)
        small=cv2.resize(dot,(50,40));y,x=np.mgrid[:40,:50]
        measured=np.array([(small*x).sum(),(small*y).sum()])/small.sum()
        np.testing.assert_allclose(measured,resize_points(center,dot.shape,small.shape),atol=.002)

    def test_regenerate_is_read_only_for_source(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);source=root/'source';out=root/'review'
            image=root/'original.png';save_image(image,np.full((100,120),128,np.uint8))
            write_json(source/'frames.json',[dict(id='frame',ids=[0],p=[[40.25,50.75]],P=[[0,0,0]])])
            write_json(source/'audit.json',[dict(file=str(image))])
            # Match audit's file stem to the saved frame ID.
            write_json(source/'frames.json',[dict(id='original',ids=[0],p=[[40.25,50.75]],P=[[0,0,0]])])
            before={p.name:p.read_bytes() for p in source.iterdir()}
            regenerate(source,out,'single',full_resolution=True)
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir()})
            self.assertTrue((out/'full_resolution/original_single.png').is_file())
            self.assertFalse(read_json(out/'review_manifest.json')['center_data_changed'])
            with self.assertRaises(ValueError):regenerate(source,out,'single')

    def test_review_cli_dialog_and_flags(self):
        with patch('bbs_calib.step_cli.choose_directory',side_effect=['source','out']) as dialog,patch('bbs_calib.step_cli.resolve_mode',return_value='prism'),patch('bbs_calib.review_images.regenerate',return_value=Path('out')) as render,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['review','--full-resolution','--image-id','frame']),0)
            self.assertEqual(dialog.call_count,2)
            render.assert_called_once_with('source','out','prism',image_id='frame',full_resolution=True)
