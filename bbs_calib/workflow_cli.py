"""CLI for rectangular full-grid real-data workflows.

Keeps the existing optical solvers unchanged. Run extraction, review its PNGs,
then explicitly confirm the ordering before fitting.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .io import new_output, read_json, write_json, serializable


def parser():
    ap=argparse.ArgumentParser(description='Rectangular dark-dot calibration; review correspondence before fitting')
    sub=ap.add_subparsers(dest='workflow',required=True)
    for name in ('single','prism'):
        p=sub.add_parser(name,help='No-prism camera' if name=='single' else 'Midpoint-split biprism camera')
        actions=p.add_subparsers(dest='action',required=True)
        e=actions.add_parser('extract',help='Extract centers and review images; does not fit')
        e.add_argument('--images',required=True,type=Path)
        e.add_argument('--output',required=True,type=Path)
        e.add_argument('--pitch-mm',type=float,default=3)
        e.add_argument('--rows',type=int,default=12)
        e.add_argument('--cols',type=int,default=9)
        if name=='prism':
            e.add_argument('--config',type=Path)
            e.add_argument('--left-suffix',default='_0')
            e.add_argument('--right-suffix',default='_1')
            e.add_argument('--pattern',default='*.tif')
            e.add_argument('--dot-profile',choices=['legacy','small','standard'],default='legacy')
            e.add_argument('--require-complete-pair',action='store_true')
        f=actions.add_parser('fit',help='Fit previously extracted, visually reviewed centers')
        f.add_argument('--source',required=True,type=Path)
        f.add_argument('--output',required=True,type=Path)
        choices=['base','fixed_center','zero_tilt'] if name=='single' else ['base','center_plus','center_minus','initial_alternative']
        f.add_argument('--variant',choices=choices,default='base')
        f.add_argument('--confirm-order',action='store_true',help='Confirm review PNG point numbering, especially left/right correspondence')
        f.add_argument('--show-jkl',action='store_true',help='Include signed J, K, L (px/mm) and save jkl.json')
        f.add_argument('--solver',choices=['enhanced','baseline'],default='enhanced',help='Default: enhanced multistart + dense refinement; baseline reproduces original single start')
        f.add_argument('--starts',type=int,help='Enhanced initializations, default 5')
        f.add_argument('--search-seed',type=int,help='Independent search seed, default 73129; does not change held-out split')
        f.add_argument('--refine-nfev',type=int,help='Final refinement evaluation budget per start, default 150')
        f.add_argument('--backend',choices=['auto','sparse','legacy'])
        f.add_argument('--workers',type=int)
        from .hardware_refinement import add_arguments
        add_arguments(f)
    p=sub.add_parser('inspect',help='Read angles, fit quality and prism pose without recalculating')
    p.add_argument('--calibration',required=True,type=Path,help='calibration.json or its containing directory')
    p.add_argument('--show-jkl',action='store_true',help='Include signed J, K, L from the saved intrinsic matrix (no refit)')
    return ap


def calibration_jkl(calibration):
    key='K_eff' if 'K_eff' in calibration else 'K'
    matrix=np.asarray(calibration[key],dtype=float)
    if matrix.shape!=(2,2) or not np.isfinite(matrix).all():
        raise ValueError('JKL requires a finite 2 x 2 intrinsic matrix')
    if abs(matrix[0,1])>1e-10:
        raise ValueError('JKL requires the lower-triangular intrinsic convention')
    return dict(J=float(matrix[0,0]),K=float(matrix[1,1]),L=float(matrix[1,0]),
                units='px/mm',matrix=matrix.tolist(),matrix_layout='[[J, 0], [L, K]]',source_field=key,
                note='Raw fitted signed coefficients; no absolute value, pixel-size conversion or mechanical certification.')


def inspect_result(path,show_jkl=False):
    from bbs_calib.angles import effective_angles
    from scipy.spatial.transform import Rotation
    path=Path(path).resolve()
    if path.is_dir():path=path/'calibration.json'
    c=read_json(path)
    result=dict(calibration_file=str(path),warning='Conditional model angles, not independently certified mechanical angles; sign ambiguity remains.')
    if show_jkl:result['jkl']=calibration_jkl(c)
    if 'K_eff' in c:
        result.update(model='biprism',status=c['status'],mechanical_angles=c.get('mechanical_angles'))
        if c['status'].get('geometry_valid'):
            result['angles']=effective_angles(c['K_eff'],c['config_snapshot']['camera']['pixel_size_um'])
            scale=result['angles']['m_px_per_mm']
            pitch=c['config_snapshot']['camera']['pixel_size_um'][0]
            result['magnification']=dict(m_px_per_mm=scale,pixel_size_um=pitch,
                magnitude_x=abs(scale)*pitch/1000.,
                status='conditional_effective_not_independently_measured')
        else:result['angles']=dict(status='rejected_invalid_optical_paths_no_angle_reported')
        omega=np.asarray(c['omega_p'],float)
        result['prism_pose']=dict(rotvec_deg=np.degrees(omega),euler_xyz_deg=Rotation.from_rotvec(omega).as_euler('xyz',degrees=True),
                                  note='Joint-model estimated prism rotation, not an independent measurement; translation is not fitted.')
        for filename in ('metrics','diagnostics'):
            f=path.parent/(filename+'.json')
            if f.exists():
                data=read_json(f)
                result[filename]=data if filename=='metrics' else dict(at_bounds=data.get('at_bounds'),
                    data_jacobian={k:data.get('data_jacobian',{}).get(k) for k in ('rank','columns')})
    elif 'K' in c and 'stages' in c:
        result.update(model='single_camera',angles=effective_angles(c['K']),variant=c.get('variant'),
                      training_rms_px=c['stages'][-1]['rms_px'],heldout_rms_px=c.get('validation',{}).get('rms_px'),
                      optimization_converged=bool(c['stages'][-1].get('success')),
                      at_bounds=c.get('at_bounds'),global_diagnostics={k:c.get('global_diagnostics',{}).get(k) for k in ('rank','columns','full_column_rank')})
    else:raise ValueError('Unrecognized calibration schema')
    if c.get('optimizer_search'):
        search=c['optimizer_search']
        result['optimizer_search']={k:search[k] for k in ('status','selected_trial','attempted','eligible_count','near_optimal_count','near_optimal_ranges','confidence_interval','global_uniqueness_proven')}
    if c.get('hardware_refinement'):
        result['hardware_refinement']=c['hardware_refinement']
    return serializable(result)


def main(argv=None):
    a=parser().parse_args(argv)
    try:
        if a.workflow=='inspect':
            print(json.dumps(inspect_result(a.calibration,show_jkl=a.show_jkl),ensure_ascii=False,indent=2));return 0
        if a.workflow=='single':from . import single_workflow as driver
        else:from . import prism_workflow as driver
        if a.action=='extract':
            from .prism_workflow import validate_grid
            validate_grid(a.rows,a.cols)
            if a.workflow=='prism' and a.config:
                from .config import load_config
                c=load_config(a.config)
                if (c['board'].get('rows'),c['board'].get('cols'),c['board'].get('pitch_mm'))!=(a.rows,a.cols,a.pitch_mm):
                    raise ValueError('--rows/--cols/--pitch-mm must match --config')
                if c['images'].get('mode')!='paired_crops' or c['images'].get('crop_layout')!='matlab_midpoint_vertical':
                    raise ValueError('Quick extract requires midpoint vertical crops; use detect --config for arbitrary ROIs/crops')
                if c['board'].get('points_csv'):
                    raise ValueError('Quick extract requires a regular grid; use detect for custom coordinates')
                if c.get('ridge_masks') or any(c['views'][v].get('roi') or c['views'][v].get('masks') for v in ('left','right')):
                    raise ValueError('Quick extract does not apply ROI/masks; use detect --config')
                if c['images'].get('crop_origins_csv') or any(c['views'][v].get('crop_origin_px') is not None for v in ('left','right')):
                    raise ValueError('Quick extract infers midpoint origins; use detect for explicit origins')
            if not np.isfinite(a.pitch_mm) or a.pitch_mm<=0:raise ValueError('--pitch-mm must be positive and finite')
            images=a.images.resolve()
            if not images.is_dir():raise ValueError('Image directory does not exist: '+str(images))
            out=new_output(a.output)
            print('Board: %d rows x %d cols, pitch %g mm. Verify the physical target.'%(a.rows,a.cols,a.pitch_mm),flush=True)
            if a.workflow=='single':driver.extract(out,directory=images,pitch_mm=a.pitch_mm,rows=a.rows,cols=a.cols)
            else:driver.extract(out,0,directory=images,pitch_mm=a.pitch_mm,pattern=a.pattern,
                                dot_profile=a.dot_profile,require_complete_pair=a.require_complete_pair,rows=a.rows,cols=a.cols,
                                config=a.config,left_suffix=a.left_suffix,right_suffix=a.right_suffix)
            write_json(out/'cli_input.json',dict(workflow=a.workflow,images=str(images),rows=a.rows,cols=a.cols,pitch_mm=a.pitch_mm,
                       point_order='image-top row and image-left column; review required'))
            print('Extraction saved: '+str(out)+'\nReview the PNGs and audit.json before fit --confirm-order.',flush=True)
            if a.workflow=='prism' and (out/'extraction_summary.json').exists() and not read_json(out/'extraction_summary.json')['retained_pairs']:
                print('WARNING: No usable complete grid pairs. Inspect audit.json; fitting cannot proceed.',file=sys.stderr)
                return 3
            if a.workflow=='single' and (out/'frames.json').exists() and not read_json(out/'frames.json'):
                print('WARNING: No usable single-camera frames.',file=sys.stderr)
                return 3
        else:
            from .solver_search import options
            solver_options=options(a.solver,a.starts,a.search_seed,a.refine_nfev)
            if a.refine_hardware is not None:
                if a.workflow!='prism':raise ValueError('--refine-hardware only supports prism')
                solver_options['refine_hardware']=a.refine_hardware
            if a.workflow=='single' and (a.backend is not None or a.workers is not None):raise ValueError('--backend/--workers currently support prism only')
            if a.workers is not None:
                if not 1<=a.workers<=32:raise ValueError('--workers must be 1–32')
                solver_options['workers']=a.workers
            if a.backend is not None:solver_options['backend']=a.backend
            if a.backend=='legacy' and (a.workers or 1)>1:raise ValueError('Parallel derivatives require auto/sparse backend')
            if not a.confirm_order:raise ValueError('Review extraction PNG numbering first, then pass --confirm-order')
            source=a.source.resolve()
            required=['frames.json','audit.json'] if a.workflow=='single' else ['config.json','observations.csv']
            for name in required:
                if not (source/name).is_file():raise ValueError('Missing extraction file: '+str(source/name))
            if (source/'cli_input.json').exists() and read_json(source/'cli_input.json')['workflow']!=a.workflow:
                raise ValueError('Extraction belongs to a different workflow')
            if a.workflow=='prism':
                from .config import load_config,validate_config
                from .hardware_refinement import apply_mode
                config=load_config(source/'config.json')
                apply_mode(config,a.refine_hardware)
                validate_config(config,'calibrate')
            out=new_output(a.output)
            write_json(out/'cli_run.json',dict(workflow=a.workflow,source=str(source),variant=a.variant,order_confirmed=True,show_jkl=a.show_jkl,solver_options=solver_options,argv=argv))
            driver.fit(source,out,a.variant,solver_options=solver_options)
            summary=inspect_result(out,show_jkl=a.show_jkl)
            write_json(out/'cli_summary.json',summary)
            if a.show_jkl:write_json(out/'jkl.json',summary['jkl'])
            print(json.dumps(summary,ensure_ascii=False,indent=2))
            converged=summary.get('optimization_converged',summary.get('status',{}).get('optimization_converged',False))
            if not converged or not summary.get('status',{}).get('geometry_valid',True):return 3
        return 0
    except (ValueError,KeyError,OSError,ImportError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
