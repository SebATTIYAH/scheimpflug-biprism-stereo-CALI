"""Small action-first adapter; all computation stays in the existing workflows."""
import argparse
from pathlib import Path
import sys


def parser():
    ap=argparse.ArgumentParser(prog='python calibrate.py',description='Stepwise calibration: extract centers, review them, fit, then inspect results')
    sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('doctor',help='Check Python and numerical dependencies without restricting environment names')
    t=sub.add_parser('init',help='Create a configuration template without overwriting an existing file')
    t.add_argument('--output',required=True)
    t.add_argument('--template',choices=['cropped','full-frame','synthetic'],default='cropped')
    e=sub.add_parser('extract',help='Step 1: extract dot centers from images without fitting a calibration')
    e.add_argument('--mode',choices=['single','prism'],required=True,help='single: one camera without a prism; prism: left/right bi-prism crops')
    e.add_argument('--images',help='Directory of original images; opens a folder dialog if omitted')
    e.add_argument('--output',help='Empty or new directory for extracted centers; opens a folder dialog if omitted')
    e.add_argument('--pitch-mm',type=float,default=3,help='Dot pitch in mm (default: 3); set grid size with --rows/--cols')
    e.add_argument('--rows',type=int,help='Grid rows (legacy default: 12)')
    e.add_argument('--cols',type=int,help='Grid columns (legacy default: 9)')
    e.add_argument('--config',help='Prism hardware configuration; grid settings must match the command')
    e.add_argument('--left-suffix',help='Left image suffix (default: _0)')
    e.add_argument('--right-suffix',help='Right image suffix (default: _1)')
    e.add_argument('--pattern',help='Prism mode only (default: *.tif)')
    e.add_argument('--dot-profile',choices=['legacy','small','standard'],help='Prism mode: standard supports filled and ring fiducials; small is for small dots; default: legacy')
    e.add_argument('--require-complete-pair',action='store_true',help='Prism mode: keep only pairs with all rows x cols accepted points in both views')
    f=sub.add_parser('fit',help='Step 2: fit calibration from reviewed dot centers; detect mode from source')
    f.add_argument('--source',help='Center directory produced by extract, not the image directory; opens a folder dialog if omitted')
    f.add_argument('--output',help='Empty or new calibration output directory; opens a folder dialog if omitted')
    f.add_argument('--mode',choices=['single','prism'],help='Optional: detect mode from source files when omitted')
    f.add_argument('--variant',default='base',choices=['base','fixed_center','zero_tilt','center_plus','center_minus','initial_alternative'])
    f.add_argument('--confirm-order',action='store_true',help='Confirm reviewed center IDs; matching left/right IDs must refer to the same physical point')
    f.add_argument('--show-jkl',action='store_true',help='Display J, K and L and save jkl.json')
    f.add_argument('--solver',choices=['enhanced','baseline'],help='enhanced (default): multiple starts and refinement; baseline: original single start')
    f.add_argument('--starts',type=int,help='Initializations in enhanced mode (default: 5)')
    f.add_argument('--search-seed',type=int,help='Search seed; does not change the train/held-out split')
    f.add_argument('--refine-nfev',type=int,help='Final refinement budget per start (default: 150)')
    f.add_argument('--backend',choices=['auto','sparse','legacy'],help='Prism solver: auto selects sparse solving for large datasets; legacy uses dense refinement')
    f.add_argument('--workers',type=int,help='Prism finite-difference worker threads (1-32; default: 1)')
    from .hardware_refinement import add_arguments
    add_arguments(f)
    p=sub.add_parser('inspect',help='Step 3: inspect saved results without refitting')
    p.add_argument('--calibration',help='Calibration directory or calibration.json; opens a folder dialog if omitted')
    p.add_argument('--show-jkl',action='store_true',help='Also display J, K and L')
    r=sub.add_parser('review',help='Regenerate center review images without changing centers or refitting')
    r.add_argument('--source',help='Existing extract output directory; opens a folder dialog if omitted')
    r.add_argument('--output',help='New empty directory for review images; opens a folder dialog if omitted')
    r.add_argument('--image-id',help='Process only this frame ID (without _0/_1 or file extension)')
    r.add_argument('--full-resolution',action='store_true',help='Also save center overlays at original image resolution')
    a=sub.add_parser('audit-points',help='Inspect high-reprojection-error points using image crops, subpixel edges, ellipses and predictions; no fitting or deletion')
    a.add_argument('--calibration',help='Calibration directory for a complete rectangular prism grid; opens a folder dialog if omitted')
    a.add_argument('--output',help='New empty directory for point diagnostics; opens a folder dialog if omitted')
    a.add_argument('--top',type=int,default=30,help='Number of largest training errors to inspect (default: 30)')
    b=sub.add_parser('filter-blur',help='Filter blurred dots by image edge width and save a new center dataset; does not refit')
    b.add_argument('--source',help='Prism extract center directory; opens a folder dialog if omitted')
    b.add_argument('--output',help='New empty directory for filtered centers; opens a folder dialog if omitted')
    b.add_argument('--max-edge-width-px',type=float,default=8.,help='Maximum 75th percentile of the 10-90% edge width (default: 8 px); lower is stricter')
    return ap


class SelectionCancelled(Exception):
    pass


def choose_directory(title,initialdir=None,mustexist=True):
    """Lazy native folder dialog; explicit CLI paths never require Tk."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise ValueError('Folder dialog unavailable in this Python environment; specify directory paths explicitly.') from exc
    root=None
    try:
        root=tk.Tk()
        root.withdraw()
        root.attributes('-topmost',True)
        root.update_idletasks()
        selected=filedialog.askdirectory(parent=root,title=title,
            initialdir=str(Path(initialdir or Path.cwd()).resolve()),mustexist=mustexist)
    except tk.TclError as exc:
        raise ValueError('Could not open the folder dialog; specify directory paths explicitly. Cause: '+str(exc)) from exc
    finally:
        if root is not None:
            try:root.destroy()
            except tk.TclError:pass
    if not selected:raise SelectionCancelled('Folder selection cancelled; no operation was started.')
    path=Path(selected).resolve()
    if mustexist and not path.is_dir():raise ValueError('Selected directory does not exist: '+str(path))
    if path.exists() and not path.is_dir():raise ValueError('Select a directory, not a file: '+str(path))
    print(title+': '+str(path),flush=True)
    return str(path)


def resolve_mode(source,requested=None):
    from .io import read_json
    source=Path(source)
    signatures={'single':('frames.json','audit.json'),'prism':('config.json','observations.csv')}
    found=[mode for mode,names in signatures.items() if all((source/name).is_file() for name in names)]
    if len(found)!=1:
        raise ValueError('Could not identify a unique center dataset. --source must point to an extract output directory and cannot mix single-camera and prism data.')
    mode=found[0]
    if requested and requested!=mode:raise ValueError('--mode does not match the center dataset; detected mode: '+mode)
    metadata=source/'cli_input.json'
    if metadata.exists() and read_json(metadata).get('workflow')!=mode:
        raise ValueError('cli_input.json does not match the center dataset; check --source.')
    return mode


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    ap=parser()
    if not argv:
        ap.print_help();return 0
    a=ap.parse_args(argv)
    try:
        if a.action=='init':
            from .templates import write_template
            print('Configuration template: '+str(write_template(a.output,a.template)))
            return 0
        if a.action=='doctor':
            from .__main__ import main as old_main
            return old_main(['doctor'])
        from .workflow_cli import main as workflow_main
        if a.action=='filter-blur':
            if not a.source:a.source=choose_directory('Select prism center directory (extract output)')
            if resolve_mode(a.source)!='prism':raise ValueError('filter-blur currently requires a complete prism grid')
            if not a.output:a.output=choose_directory('Select or create an empty directory for filtered centers',initialdir=Path(a.source).resolve().parent,mustexist=False)
            from .blur_filter import filter_dataset
            out,summary=filter_dataset(a.source,a.output,a.max_edge_width_px)
            print('Filtered centers saved: '+str(out)+'; categories: '+str(summary['counts']),flush=True)
            return 0 if summary['fit_ready'] else 3
        if a.action=='audit-points':
            if a.top<1:raise ValueError('--top must be a positive integer')
            if not a.calibration:a.calibration=choose_directory('Select prism calibration directory (contains calibration.json)')
            if not a.output:a.output=choose_directory('Select or create an empty directory for point diagnostics',initialdir=Path(a.calibration).resolve().parent,mustexist=False)
            from .point_audit import generate
            out=generate(a.calibration,a.output,a.top)
            print('Diagnostics saved without changing centers or calibration: '+str(out),flush=True)
            return 0
        if a.action=='review':
            if not a.source:a.source=choose_directory('Select existing center extraction directory')
            mode=resolve_mode(a.source)
            if not a.output:a.output=choose_directory('Select or create an empty directory for review images',initialdir=Path(a.source).resolve().parent,mustexist=False)
            from .review_images import regenerate
            out=regenerate(a.source,a.output,mode,image_id=a.image_id,full_resolution=a.full_resolution)
            print('Review images regenerated without changing center data: '+str(out),flush=True)
            return 0
        if a.action=='inspect':
            if not a.calibration:
                a.calibration=choose_directory('Select calibration directory (contains calibration.json)')
            args=['inspect','--calibration',a.calibration]
            if a.show_jkl:args.append('--show-jkl')
            return workflow_main(args)
        if a.action=='extract':
            if a.mode=='single' and any(v is not None for v in (a.config,a.left_suffix,a.right_suffix)):
                raise ValueError('--config/--left-suffix/--right-suffix are only valid in prism mode')
            if a.mode=='single' and (a.dot_profile is not None or a.require_complete_pair):
                raise ValueError('--dot-profile/--require-complete-pair are only valid in prism mode')
            if a.mode=='single' and a.pattern is not None:
                raise ValueError('--pattern is only valid in prism mode; single mode reads BMP/TIF/TIFF/PNG/JPG automatically.')
            if not a.images:
                a.images=choose_directory('Select original image directory: '+('single-camera images' if a.mode=='single' else 'paired prism _0 and _1 images'))
            if not a.output:
                a.output=choose_directory('Select or create an empty directory for centers and review images',initialdir=Path(a.images).resolve().parent,mustexist=False)
            args=[a.mode,'extract','--images',a.images,'--output',a.output,'--pitch-mm',str(a.pitch_mm)]
            if a.pattern is not None:args+=['--pattern',a.pattern]
            if a.dot_profile is not None:args+=['--dot-profile',a.dot_profile]
            if a.require_complete_pair:args.append('--require-complete-pair')
            for flag,value in [('--rows',a.rows),('--cols',a.cols),('--config',a.config),('--left-suffix',a.left_suffix),('--right-suffix',a.right_suffix)]:
                if value is not None:args.extend([flag,str(value)])
        else:
            if not a.source:
                a.source=choose_directory('Select extracted center directory (not original images)')
            mode=resolve_mode(a.source,a.mode)
            if mode=='single' and a.refine_hardware is not None:
                raise ValueError('--refine-hardware only supports prism')
            variants=['base','fixed_center','zero_tilt'] if mode=='single' else ['base','center_plus','center_minus','initial_alternative']
            if a.variant not in variants:raise ValueError('Supported --variant values for '+mode+': '+', '.join(variants))
            if not a.output:
                a.output=choose_directory('Select or create an empty directory for this calibration',initialdir=Path(a.source).resolve().parent,mustexist=False)
            args=[mode,'fit','--source',a.source,'--output',a.output,'--variant',a.variant]
            if a.confirm_order:args.append('--confirm-order')
            if a.show_jkl:args.append('--show-jkl')
            print('Calibration mode: '+mode+'; center directory: '+str(Path(a.source).resolve()),flush=True)
            for flag,value in [('--solver',a.solver),('--starts',a.starts),('--search-seed',a.search_seed),('--refine-nfev',a.refine_nfev)]:
                if value is not None:args += [flag,str(value)]
            for flag,value in [('--backend',a.backend),('--workers',a.workers)]:
                if value is not None:args += [flag,str(value)]
            if a.refine_hardware is not None:args += ['--refine-hardware',a.refine_hardware]
        return workflow_main(args)
    except SelectionCancelled as exc:
        print(str(exc),file=sys.stderr);return 130
    except (ValueError,KeyError,OSError,ImportError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr);return 2
