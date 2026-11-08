import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

from bbs_calib.angles import effective_angles
from bbs_calib.config import validate_config
from bbs_calib.io import read_json, write_json
from bbs_calib.prism_workflow import find_grid, pair_complete, validate_grid
from bbs_calib.step_cli import main
from bbs_calib.templates import template_config, write_template


class PublicReleaseTests(unittest.TestCase):
    def test_arbitrary_rectangular_grids(self):
        for rows, cols in ((6,9),(7,5),(5,6)):
            gray=np.full((rows*24+70,cols*24+70),220,np.uint8)
            for r in range(rows):
                for c in range(cols):cv2.circle(gray,(35+c*24,35+r*24),4,20,-1)
            blobs,selected,_=find_grid(gray,rows=rows,cols=cols,min_area_px=8)
            self.assertIsNotNone(selected)
            self.assertEqual(len(set(selected)),rows*cols)
            np.testing.assert_allclose(blobs[selected[-1]][0],[35+(cols-1)*24,35+(rows-1)*24],atol=.1)

    def test_complete_pair_variable_size(self):
        rows=[dict(view=v,point_id=str(i)) for v in ('left','right') for i in range(30)]
        self.assertTrue(pair_complete(rows,5,6))
        self.assertFalse(pair_complete(rows[:-1],5,6))

    def test_bad_grid_dimensions(self):
        for shape in ((1,9),(2,2),(3.5,9),(True,9)):
            with self.assertRaises(ValueError):validate_grid(*shape)

    def test_template_has_no_experiment_hardware(self):
        c=template_config('cropped')
        for key in ('refractive_index','wedge_L_deg','wedge_R_deg','thickness_mm'):
            self.assertIsNone(c['prism'][key])
        self.assertIsNone(c['camera']['pixel_size_um'])
        self.assertIsNone(c['camera']['m_px_per_mm'])
        with self.assertRaisesRegex(ValueError,'Missing required'):validate_config(c)

    def test_template_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'hardware.json'
            write_template(path);before=path.read_bytes()
            with self.assertRaises(FileExistsError):write_template(path)
            self.assertEqual(path.read_bytes(),before)

    def test_no_environment_name_gate(self):
        from bbs_calib.__main__ import main as dispatch
        with patch('sys.prefix','unrelated-environment'),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(dispatch(['doctor']),0)
            self.assertTrue(json.loads(output.getvalue())['dependencies_imported'])

    def test_module_action_dispatch(self):
        from bbs_calib.__main__ import main as dispatch
        with patch('bbs_calib.step_cli.main',return_value=9) as action:
            self.assertEqual(dispatch(['extract','--help']),9)
            action.assert_called_once_with(['extract','--help'])

    def test_cli_forwards_generic_parameters(self):
        with patch('bbs_calib.workflow_cli.main',return_value=0) as driver:
            self.assertEqual(main(['extract','--mode','prism','--images','in','--output','out',
                '--rows','6','--cols','9','--left-suffix','_L','--right-suffix','_R','--config','hardware.json']),0)
            args=driver.call_args.args[0]
            self.assertEqual(args[args.index('--rows')+1],'6')
            self.assertEqual(args[args.index('--left-suffix')+1],'_L')

    def test_missing_hardware_does_not_create_fit_output(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stderr(io.StringIO()):
            root=Path(tmp);source=root/'centers';source.mkdir()
            write_template(source/'config.json')
            (source/'observations.csv').touch()
            self.assertEqual(main(['fit','--source',str(source),'--output',str(root/'fit'),'--confirm-order']),2)
            self.assertFalse((root/'fit').exists())

    def test_reject_nonfinite_hardware(self):
        for key in ('refractive_index','wedge_L_deg','thickness_mm'):
            c=template_config('synthetic');c['prism'][key]=float('nan')
            with self.assertRaises(ValueError):validate_config(c)
        for pixel in (None,[1],[0,1],[1,float('inf')]):
            c=template_config('synthetic');c['camera']['pixel_size_um']=pixel
            with self.assertRaises(ValueError):validate_config(c)

    def test_effective_angles_bad_pixel_geometry(self):
        for pixel in ([0,1],[1],[float('nan'),1]):
            with self.assertRaises(ValueError):effective_angles([[1,0],[0,1]],pixel)

    def test_small_blur_rejected_before_output(self):
        from bbs_calib.blur_filter import filter_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=template_config('cropped');c['full_grid_extraction']={'dot_profile':'small'}
            write_json(root/'source/config.json',c)
            with self.assertRaisesRegex(ValueError,'Small-dot blur'):filter_dataset(root/'source',root/'filtered')
            self.assertFalse((root/'filtered').exists())

    def test_runtime_respects_user_threads(self):
        from bbs_calib.runtime import configure
        import os
        with patch.dict(os.environ,{'OPENBLAS_NUM_THREADS':'2'}):
            configure();self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'],'2')

    def test_export_is_allowlisted_and_non_overwriting(self):
        from tools.export_public import export
        with tempfile.TemporaryDirectory() as tmp:
            path,count=export(Path(tmp)/'source.zip')
            with zipfile.ZipFile(path) as archive:
                names=archive.namelist()
                self.assertIn('bbs_calib/resources/synthetic.json',names)
                self.assertIn('pyproject.toml',names)
                self.assertIn('extensions/brown_v4/joint_calibrate.py',names)
                self.assertNotIn('tools/private_calibration.py',names)
                self.assertFalse(any(n.startswith(('outputs/','private_data/')) for n in names))
                self.assertFalse(any(n.lower().endswith(('.tif','.pdf','.png')) for n in names))
                manifest=json.loads(archive.read('SOURCE_MANIFEST.json'))
                self.assertEqual(manifest['schema_version'],1)
                self.assertEqual(len(manifest['files_sha256']),count)
                self.assertIsNone(json.loads(archive.read('configs/cropped.example.json'))['prism']['thickness_mm'])
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):export(path)
            self.assertEqual(before,path.read_bytes())
