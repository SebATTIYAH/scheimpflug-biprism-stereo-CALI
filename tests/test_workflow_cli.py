import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bbs_calib.workflow_cli import main, parser, inspect_result, calibration_jkl
from bbs_calib.io import write_json


class WorkflowCLITests(unittest.TestCase):
    def test_jkl_signed_mapping_both_models(self):
        for key in ('K','K_eff'):
            r=calibration_jkl({key:[[-29.8,0],[.067,-30.2]]})
            self.assertEqual((r['J'],r['K'],r['L']),(-29.8,-30.2,.067))
            self.assertEqual(r['units'],'px/mm')
            self.assertEqual(r['source_field'],key)

    def test_jkl_rejects_invalid_matrix(self):
        for matrix in ([[1,2],[3,4]],[[1,0],[0,float('nan')]],[[1,0,0],[0,1,0]]):
            with self.assertRaises(ValueError):calibration_jkl({'K':matrix})

    def test_jkl_option_on_both_fit_parsers(self):
        for name in ('single','prism'):
            a=parser().parse_args([name,'fit','--source','input','--output','output','--show-jkl'])
            self.assertTrue(a.show_jkl)

    def test_inspect_jkl_opt_in_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'calibration.json'
            write_json(path,dict(K=[[-30,0],[.1,-31]],stages=[dict(success=True,rms_px=.1)]))
            original=path.read_bytes()
            self.assertNotIn('jkl',inspect_result(tmp))
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(['inspect','--calibration',tmp,'--show-jkl']),0)
            self.assertEqual(json.loads(output.getvalue())['jkl']['L'],.1)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(len(list(Path(tmp).iterdir())),1)

    def test_fit_saves_jkl_only_when_requested(self):
        for show in (False,True):
            with tempfile.TemporaryDirectory() as tmp,patch('sys.prefix','test_environment'),contextlib.redirect_stdout(io.StringIO()):
                source=Path(tmp)/'source';out=Path(tmp)/'result'
                write_json(source/'frames.json',[]);write_json(source/'audit.json',[])
                def fake_fit(source,out,variant,**kwargs):
                    write_json(out/'calibration.json',dict(K=[[-30,0],[.1,-31]],stages=[dict(success=True,rms_px=.1)]))
                args=['single','fit','--source',str(source),'--output',str(out),'--confirm-order']
                if show:args.append('--show-jkl')
                with patch('bbs_calib.single_workflow.fit',side_effect=fake_fit):self.assertEqual(main(args),0)
                self.assertEqual((out/'jkl.json').exists(),show)
                summary=json.loads((out/'cli_summary.json').read_text(encoding='utf-8'))
                self.assertEqual('jkl' in summary,show)
                if show:self.assertEqual(json.loads((out/'jkl.json').read_text(encoding='utf-8')),summary['jkl'])

    def test_parser_single(self):
        a=parser().parse_args(['single','extract','--images','input','--output','output'])
        self.assertEqual(a.pitch_mm,3)

    def test_parser_prism(self):
        a=parser().parse_args(['prism','fit','--source','input','--output','output','--variant','center_plus','--confirm-order'])
        self.assertTrue(a.confirm_order)

    def test_confirmation_required_without_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sys.prefix','test_environment'), contextlib.redirect_stderr(io.StringIO()):
            out=Path(tmp)/'result'
            self.assertEqual(main(['single','fit','--source',tmp,'--output',str(out)]),2)
            self.assertFalse(out.exists())

    def test_nonempty_output_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sys.prefix','test_environment'), contextlib.redirect_stderr(io.StringIO()):
            root=Path(tmp);write_json(root/'keep.json',{'x':1})
            self.assertEqual(main(['single','extract','--images',tmp,'--output',tmp]),2)
            self.assertEqual((root/'keep.json').read_text().strip(),'{\n  "x": 1\n}')

    def test_invalid_pitch(self):
        with contextlib.redirect_stderr(io.StringIO()),patch('sys.prefix','test_environment'):
            self.assertEqual(main(['single','extract','--images','.','--output','unused','--pitch-mm','nan']),2)

    def test_single_inspect(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp)/'calibration.json',dict(K=[[-30,0],[0,-30]],stages=[dict(success=True,rms_px=.1)]))
            r=inspect_result(tmp)
            self.assertEqual(r['angles']['total_tilt_deg'],0)
            self.assertTrue(r['optimization_converged'])

    def test_invalid_prism_paths_suppress_angles(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp)/'calibration.json',dict(K_eff=[[-30,0],[0,-31]],omega_p=[0,0,0],status=dict(geometry_valid=False)))
            r=inspect_result(tmp)
            self.assertNotIn('total_tilt_deg',r['angles'])
            self.assertEqual(r['prism_pose']['rotvec_deg'],[0,0,0])

    def test_custom_extract_routes_only_requested_directory(self):
        with tempfile.TemporaryDirectory() as tmp,patch('sys.prefix','test_environment'),patch('bbs_calib.single_workflow.extract') as extract,contextlib.redirect_stdout(io.StringIO()):
            out=Path(tmp)/'extract'
            self.assertEqual(main(['single','extract','--images',tmp,'--output',str(out),'--pitch-mm','4']),0)
            self.assertEqual(extract.call_args.kwargs['directory'],Path(tmp).resolve())
            self.assertEqual(extract.call_args.kwargs['pitch_mm'],4)

    def test_main_dispatch(self):
        from bbs_calib.__main__ import main as package_main
        with patch('bbs_calib.workflow_cli.main',return_value=7) as workflow:
            self.assertEqual(package_main(['single','--help']),7)
            workflow.assert_called_once_with(['single','--help'])
