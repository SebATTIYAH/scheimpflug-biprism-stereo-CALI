import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bbs_calib.io import write_json,write_csv
from bbs_calib.step_cli import main,resolve_mode,choose_directory,SelectionCancelled


class StepCLITests(unittest.TestCase):
    def test_missing_extract_paths_open_two_dialogs(self):
        with patch('bbs_calib.step_cli.choose_directory',side_effect=['images','centers']) as dialog,patch('bbs_calib.workflow_cli.main',return_value=0) as driver:
            self.assertEqual(main(['extract','--mode','prism']),0)
            self.assertEqual(dialog.call_count,2)
            self.assertFalse(dialog.call_args.kwargs['mustexist'])
            self.assertEqual(driver.call_args.args[0],['prism','extract','--images','images','--output','centers','--pitch-mm','3'])

    def test_explicit_paths_never_open_dialog(self):
        with patch('bbs_calib.step_cli.choose_directory') as dialog,patch('bbs_calib.workflow_cli.main',return_value=0):
            self.assertEqual(main(['extract','--mode','single','--images','in','--output','out']),0)
            self.assertEqual(main(['inspect','--calibration','out']),0)
            dialog.assert_not_called()

    def test_only_missing_path_opens_dialog(self):
        with patch('bbs_calib.step_cli.choose_directory',return_value='out') as dialog,patch('bbs_calib.workflow_cli.main',return_value=0) as driver:
            self.assertEqual(main(['extract','--mode','single','--images','in']),0)
            self.assertEqual(dialog.call_count,1)
            self.assertEqual(driver.call_args.args[0][3],'in')

    def test_inspect_folder_dialog(self):
        with patch('bbs_calib.step_cli.choose_directory',return_value='result') as dialog,patch('bbs_calib.workflow_cli.main',return_value=0) as driver:
            self.assertEqual(main(['inspect','--show-jkl']),0)
            dialog.assert_called_once()
            driver.assert_called_once_with(['inspect','--calibration','result','--show-jkl'])

    def test_fit_missing_paths_and_auto_mode(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            write_json(Path(tmp)/'frames.json',[]);write_json(Path(tmp)/'audit.json',[])
            with patch('bbs_calib.step_cli.choose_directory',side_effect=[tmp,'out']) as dialog,patch('bbs_calib.workflow_cli.main',return_value=0) as driver:
                self.assertEqual(main(['fit','--confirm-order','--show-jkl']),0)
                self.assertEqual(dialog.call_count,2)
                self.assertEqual(driver.call_args.args[0],['single','fit','--source',tmp,'--output','out','--variant','base','--confirm-order','--show-jkl'])

    def test_cancel_does_not_start_workflow(self):
        for selection in ([SelectionCancelled('cancel')],['in',SelectionCancelled('cancel')]):
            with patch('bbs_calib.step_cli.choose_directory',side_effect=selection),patch('bbs_calib.workflow_cli.main') as driver,contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['extract','--mode','single']),130)
                driver.assert_not_called()

    def test_tk_dialog_cancel_destroys_root(self):
        with patch('tkinter.Tk') as tkroot,patch('tkinter.filedialog.askdirectory',return_value=''):
            with self.assertRaises(SelectionCancelled):choose_directory('test')
            tkroot.return_value.destroy.assert_called_once()

    def test_tk_unavailable_shows_cli_fallback(self):
        import tkinter
        with patch('tkinter.Tk',side_effect=tkinter.TclError('display unavailable')):
            with self.assertRaisesRegex(ValueError,'specify directory paths explicitly'):choose_directory('test')

    def test_nonempty_selected_output_remains_protected(self):
        with tempfile.TemporaryDirectory() as tmp,patch('sys.prefix','test_environment'),contextlib.redirect_stderr(io.StringIO()):
            sentinel=Path(tmp)/'keep.json';write_json(sentinel,{'keep':True});original=sentinel.read_bytes()
            with patch('bbs_calib.step_cli.choose_directory',side_effect=[tmp,tmp]):
                self.assertEqual(main(['extract','--mode','single']),2)
            self.assertEqual(sentinel.read_bytes(),original)

    def test_no_arguments_shows_help(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:self.assertEqual(main([]),0)
        self.assertIn('python calibrate.py',out.getvalue())

    def test_extract_action_first_routes_both_models(self):
        for mode in ('single','prism'):
            with patch('bbs_calib.workflow_cli.main',return_value=0) as driver:
                self.assertEqual(main(['extract','--mode',mode,'--images','sample images','--output','out']),0)
                self.assertEqual(driver.call_args.args[0],[mode,'extract','--images','sample images','--output','out','--pitch-mm','3'])

    def test_inspect_forwards_show_jkl(self):
        with patch('bbs_calib.workflow_cli.main',return_value=0) as driver:
            self.assertEqual(main(['inspect','--calibration','out','--show-jkl']),0)
            driver.assert_called_once_with(['inspect','--calibration','out','--show-jkl'])

    def test_fit_auto_mode_and_exit_code(self):
        for mode in ('single','prism'):
            with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
                source=Path(tmp)
                if mode=='single':
                    write_json(source/'frames.json',[]);write_json(source/'audit.json',[])
                else:
                    write_json(source/'config.json',{});write_csv(source/'observations.csv',[])
                with patch('bbs_calib.workflow_cli.main',return_value=3) as driver:
                    self.assertEqual(main(['fit','--source',tmp,'--output','out','--confirm-order','--show-jkl']),3)
                    self.assertEqual(driver.call_args.args[0],[mode,'fit','--source',tmp,'--output','out','--variant','base','--confirm-order','--show-jkl'])

    def test_reject_inconsistent_or_ambiguous_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)
            with self.assertRaises(ValueError):resolve_mode(source)
            write_json(source/'frames.json',[]);write_json(source/'audit.json',[])
            with self.assertRaises(ValueError):resolve_mode(source,'prism')
            write_json(source/'cli_input.json',{'workflow':'prism'})
            with self.assertRaises(ValueError):resolve_mode(source)
            write_json(source/'config.json',{});write_csv(source/'observations.csv',[])
            with self.assertRaises(ValueError):resolve_mode(source)

    def test_reject_wrong_variant_before_driver(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stderr(io.StringIO()):
            write_json(Path(tmp)/'frames.json',[]);write_json(Path(tmp)/'audit.json',[])
            with patch('bbs_calib.workflow_cli.main') as driver:
                self.assertEqual(main(['fit','--source',tmp,'--output','out','--variant','center_plus']),2)
                driver.assert_not_called()

    def test_single_does_not_silently_ignore_pattern(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['extract','--mode','single','--images','in','--output','out','--pattern','*.bmp']),2)
