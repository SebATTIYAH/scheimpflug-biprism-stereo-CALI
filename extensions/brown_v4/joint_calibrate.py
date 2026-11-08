"""Separate CLI for gauge-fixed translated biprism bundle adjustment."""
import argparse
from pathlib import Path
from copy import deepcopy
import hashlib
import numpy as np
from bbs_calib.io import read_json,write_json,new_output
from bbs_calib.config import load_config,validate_config
from bbs_calib.optimization import observations,array_data
from bbs_calib.joint_bundle import (SUPPORTED_MODELS,model_for_state,validate_model,fit_once,data_rank,validate_frames,
                                   feasible_offset_interval,translated_system,project,distortion_identifiability)
from bbs_calib.brown_distortion import has_brown,parameters as brown_parameters


def radial_initial_state(state,order=None):
    """Preserve the initial order unless explicitly changed; no silent truncation."""
    from bbs_calib.projection import radial_coefficients
    base=deepcopy(state)
    for k in ('K','c','omega','kappa'):base[k]=np.asarray(base[k],float)
    k=radial_coefficients(base['kappa'])
    order=len(k) if order is None else order
    if order not in (2,3):raise ValueError('Radial order must be 2 or 3')
    if order<len(k):raise ValueError('Cannot silently discard kappa3; use a two-coefficient initial calibration for order 2')
    base['kappa']=np.pad(k,(0,order-len(k)))
    base['transverse_offset_mm']=float(base.get('transverse_offset_mm',0.))
    return base


def brown_initial_state(state,enabled=False):
    base=deepcopy(state)
    if enabled or has_brown(base):
        base.setdefault('tangential',np.zeros(2))
        base.setdefault('distortion_center_px',np.asarray(base['c'],float).copy())
        for name in ('tangential','distortion_center_px'):base[name]=np.asarray(base[name],float)
        brown_parameters(base)
    return base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='Reviewed full-frame extract directory')
    p.add_argument('--initial-calibration',required=True,help='Valid old calibration supplying full optical state')
    p.add_argument('--split-from',required=True,help='Calibration JSON whose whole-frame validation split is preserved')
    p.add_argument('--output',required=True)
    p.add_argument('--starts',type=int,default=3)
    p.add_argument('--max-nfev',type=int,default=400)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--offset-bound-mm',type=float,default=10.)
    p.add_argument('--path-weight',type=float,default=540.)
    p.add_argument('--radial-order',type=int,choices=[2,3],default=None,
                   help='Number of radial terms; 3 adds kappa3*rho^6. Default: preserve initial calibration order')
    p.add_argument('--tangential',action='store_true',help='Enable Brown p1,p2 (preserved automatically for v4 initial files)')
    center=p.add_mutually_exclusive_group()
    center.add_argument('--free-distortion-center',dest='free_center',action='store_true',default=None)
    center.add_argument('--fix-distortion-center',dest='free_center',action='store_false')
    p.add_argument('--distortion-center-bound-px',type=float,default=300.,
                   help='Each center coordinate stays within +/- this many px of fixed c; search bound, not measurement')
    p.add_argument('--tangential-bound',type=float,default=.05,help='Absolute bound for each dimensionless p1,p2')
    p.add_argument('--confirm-order',action='store_true')
    args=p.parse_args()
    if not args.confirm_order:p.error('Review physical IDs first and pass --confirm-order')
    if args.starts<1 or args.max_nfev<1 or not 1<=args.workers<=32:p.error('Invalid starts, iterations or workers')
    if not np.isfinite([args.offset_bound_mm,args.path_weight]).all() or min(args.offset_bound_mm,args.path_weight)<=0:p.error('Positive finite bounds/weight required')
    if not np.isfinite([args.distortion_center_bound_px,args.tangential_bound]).all() or min(args.distortion_center_bound_px,args.tangential_bound)<=0:p.error('Positive finite Brown bounds required')
    source=Path(args.source);c=load_config(source/'config.json');validate_config(c,'calibrate')
    obs=observations(source/'observations.csv',c)
    saved=read_json(args.initial_calibration);split=read_json(args.split_from)
    if not saved.get('status',{}).get('geometry_valid'):p.error('Initial calibration must have valid geometry')
    if saved.get('model_version') not in ('bbs-shared-ray-radial-v1',)+SUPPORTED_MODELS:p.error('Expected saved v1/v2/v3/v4 initialization')
    if saved['model_version'] in SUPPORTED_MODELS:validate_model(saved)
    elif len(saved['state']['kappa'])!=2:p.error('v1 initialization must contain two radial coefficients')
    if saved['view_branches']!={v:c['views'][v]['branch'] for v in ('left','right')}:p.error('Branch mapping differs')
    for k in ('refractive_index','wedge_L_deg','wedge_R_deg','thickness_mm'):
        if not np.isclose(c['prism'][k],saved['state']['prism'][k]):p.error('Hardware differs: '+k)
    validation=split['validation_frames']
    if not set(validation)<=set(obs['frames']):p.error('Validation split contains unavailable frames')
    training=[f for f in obs['frames'] if f not in validation]
    if len(training)<4 or not validation:p.error('Need >=4 training frames and nonempty held-out set')
    try:base=radial_initial_state(saved['state'],args.radial_order)
    except ValueError as exc:p.error(str(exc))
    base=brown_initial_state(base,args.tangential or args.free_center is True)
    free_center=(saved.get('config_snapshot',{}).get('optimization',{}).get('release_distortion_center',False)
                 if args.free_center is None else args.free_center)
    model=model_for_state(base)
    c['camera'].update(q_scale_mm=base['q_scale'],radius_limit_mm=base['radius_limit_mm'],c_px=base['c'].tolist())
    c['optimization'].update(max_nfev=args.max_nfev,workers=args.workers,
        transverse_offset_bound_mm=args.offset_bound_mm,path_weight_px_per_mm=args.path_weight,
        radial_order=len(base['kappa']),release_distortion_center=free_center,
        distortion_center_bound_px=args.distortion_center_bound_px,tangential_bound=args.tangential_bound)
    print('MODEL',model,'radial terms',len(base['kappa']),'tangential',has_brown(base),
          'free distortion center',bool(free_center),'fixed pixel origin',base['c'].tolist(),flush=True)
    data=array_data(obs['used'],training);interval=feasible_offset_interval(data,base)
    lo=max(interval[0],-args.offset_bound_mm);hi=min(interval[1],args.offset_bound_mm)
    if lo>=hi:p.error('No initial feasible offset interval; inspect mapping, bounds and optical state')
    out=new_output(args.output)
    write_json(out/'run.json',dict(arguments=vars(args),training_frames=training,validation_frames=validation,
        initial_feasible_interval_mm=interval,interval_is_not_measurement=True,
        model_version=model,radial_order=len(base['kappa']),
        source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
         for n in ['joint_calibrate.py','bbs_calib/joint_bundle.py','bbs_calib/geometry.py','bbs_calib/sparse_solver.py',
                   'bbs_calib/projection.py','bbs_calib/reconstruction.py','bbs_calib/brown_distortion.py']},
        source_observations_sha256=hashlib.sha256((source/'observations.csv').read_bytes()).hexdigest(),
        initial_calibration_sha256=hashlib.sha256(Path(args.initial_calibration).read_bytes()).hexdigest(),
        split_sha256=hashlib.sha256(Path(args.split_from).read_bytes()).hexdigest()))
    write_json(out/'config_snapshot.json',c);rng=np.random.default_rng(73129);candidates=[];best=None
    fractions=np.linspace(.2,.8,args.starts) if args.starts>1 else [.5]
    for i,fraction in enumerate(fractions):
        state=deepcopy(base);state['transverse_offset_mm']=lo+(hi-lo)*fraction
        if i==0 and saved['model_version'] in SUPPORTED_MODELS and lo<base['transverse_offset_mm']<hi:
            state['transverse_offset_mm']=base['transverse_offset_mm']
        if i:
            state['K'][0,0]*=np.exp(rng.normal(0,.003));state['K'][1,1]*=np.exp(rng.normal(0,.003))
            state['omega']+=rng.normal(0,np.radians(.1),3)
        print('JOINT BA start %d/%d offset=%.6f mm'%(i+1,args.starts,state['transverse_offset_mm']),flush=True)
        try:
            fitted,poses,d,internal=fit_once(data,state,c,'joint start %d'%(i+1))
            d['trial']=i;d['initial_offset_mm']=state['transverse_offset_mm'];candidates.append(d)
            key=(not(d['geometry_valid'] and d['distortion_invertible']),not d['success'],d['objective'])
            if best is None or key<best[0]:best=(key,fitted,poses,d,internal)
            print('RESULT',i,d['training']['euclidean_rms_px'],'geometry',d['geometry_valid'],
                  'offset',d['transverse_offset_mm'],'optimality',d['optimality'],flush=True)
        except ValueError as exc:candidates.append(dict(trial=i,error=str(exc),success=False,geometry_valid=False))
        write_json(out/'candidates.json',candidates)
    if best is None:raise ValueError('All starts failed; inspect candidates.json')
    _,state,poses,d,internal=best
    rank=data_rank(*internal);validation_result=validate_frames(obs,validation,state,c)
    distortion_id=distortion_identifiability(rank,state)
    gd=rank['global_after_eliminating_poses']
    offset_correlations=[r for r in gd['high_column_correlations']
                         if 'transverse_offset_mm' in (r['a'],r['b'])]
    weak_offset=bool(offset_correlations) or not gd['full_column_rank']
    # Training measured-point triangulation is a separate check from fitted poses.
    train_check=validate_frames(obs,training,state,c)
    eligible=[a for a in candidates if a.get('success') and a.get('geometry_valid') and a.get('distortion_invertible')]
    minimum=min([a['objective'] for a in eligible],default=np.inf)
    near=[a for a in eligible if a['objective']<=minimum+max(.01*abs(minimum),1e-8)]
    ranges={k:[min(a[k] for a in near),max(a[k] for a in near)] for k in ['transverse_offset_mm'] if near}
    status=dict(optimization_converged=d['success'],training_fitted_paths_valid=d['geometry_valid'],
        training_measured_paths_valid=train_check['all_geometry_valid'],
        held_out_paths_valid=validation_result['all_geometry_valid'],distortion_invertible=d['distortion_invertible'],
        geometry_valid=d['geometry_valid'] and train_check['all_geometry_valid'] and validation_result['all_geometry_valid'],
        data_rank_full=rank['global_after_eliminating_poses']['full_column_rank'],
        distortion_weakly_identified=distortion_id['weakly_identified'] if distortion_id else False,
        transverse_offset_weakly_identified=weak_offset,
        external_validation_available=False,global_optimum_proven=False,mechanical_position_certified=False)
    cal=dict(model_version=model,radial_order=len(state['kappa']),state=state,view_branches=saved['view_branches'],input_mode='full_frame',
        status=status,config_snapshot=c,training_frames=training,validation_frames=validation,
        extrinsics=dict(zip(training,poses)),selected_trial=d['trial'],
        gauge=dict(board_geometry_fixed=True,affine_pixel_origin_fixed=True,affine_pixel_origin_px=state['c'],
            distortion_center_px=state.get('distortion_center_px',state['c']),distortion_center_fixed=not free_center,
            along_projected_ridge_translation_mm=0.,camera_axial_translation_mm=0.,
            note='Fixed gauges; offset is conditional on camera/distortion and fixed prism hardware, not unique absolute mechanical metrology'),
        optimization=dict(eligible_count=len(eligible),near_optimal_count=len(near),near_optimal_ranges=ranges,
            selection='Training objective only, valid geometry and convergence preferred; held-out data never selects candidate'),
        position_identifiability=dict(weakly_identified=weak_offset,high_sensitivity_correlations=offset_correlations,
            warning='Multistart agreement is not position accuracy. Profile the offset with all other variables reoptimized; do not report a mechanical confidence interval.'),
        distortion_identifiability=distortion_id,distortion_certificate=d.get('distortion_certificate'),
        compatibility='Requires matching joint_bundle adapter. Brown v4 includes tangential terms and a separate distortion center; never truncate parameters or bypass version checks.')
    write_json(out/'calibration.json',cal);write_json(out/'rank_diagnostics.json',rank)
    write_json(out/'validation.json',validation_result);write_json(out/'training_measured_geometry.json',train_check)
    write_json(out/'summary.json',dict(status=status,training=d['training'],held_out=validation_result['metrics'],
        transverse_offset_mm=state['transverse_offset_mm'],near_optimal_offset_range_mm=ranges,
        selected_trial=d['trial'],state=state))
    # Curved ridge image locus from physical local x=0,z=t, not target-gap midpoint.
    system=translated_system(state);t=state['prism']['thickness_mm']
    ridge=np.column_stack([np.zeros(101),np.linspace(-40,40,101),np.full(101,t)])
    write_json(out/'ridge_projection.json',dict(local_prism_points_mm=ridge,
        pixels_left=project(ridge,'L',state,system),pixels_right=project(ridge,'R',state,system),
        warning='Model-predicted locus, not an independently detected edge.'))
    print('SAVED',out,'STATUS',status,flush=True)


if __name__=='__main__':main()
