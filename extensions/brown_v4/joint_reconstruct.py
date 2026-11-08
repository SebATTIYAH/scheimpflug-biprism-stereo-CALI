"""Reconstruct Nx4 stereo matches with translated v2, radial3 v3 or Brown v4."""
import argparse
from pathlib import Path
import numpy as np
from bbs_calib.io import read_json
from bbs_calib.joint_bundle import validate_model,reconstruct
from bbs_calib.brown_distortion import has_brown,certificate


def load_calibration(path):
    cal=read_json(path)
    validate_model(cal)
    if not cal.get('status',{}).get('geometry_valid') or not cal['status'].get('optimization_converged'):
        raise ValueError('Calibration must pass optimization and geometry checks')
    state=cal['state']
    for k,shape in [('K',(2,2)),('c',(2,)),('omega',(3,)),('kappa',(len(state['kappa']),))]:
        state[k]=np.asarray(state[k],float)
        if state[k].shape!=shape or not np.isfinite(state[k]).all():raise ValueError('Invalid '+k)
    for k in ('q_scale','radius_limit_mm'):
        if not np.isfinite(state[k]) or state[k]<=0:raise ValueError('Invalid '+k)
    if not np.isfinite(state['transverse_offset_mm']):raise ValueError('Invalid offset')
    if has_brown(state):
        for k in ('tangential','distortion_center_px'):state[k]=np.asarray(state[k],float)
        if not certificate(state)['certified']:raise ValueError('Brown field lacks an invertibility certificate')
    return cal


def reconstruct_matches(pixels,cal,pixel_coordinates='full_frame',left_origin=None,right_origin=None):
    p=np.asarray(pixels,float)
    if p.ndim!=2 or p.shape[1]!=4:raise ValueError('Expected Nx4: u_left,v_left,u_right,v_right')
    if pixel_coordinates=='full_frame':
        if left_origin is not None or right_origin is not None:raise ValueError('Do not add origins to full-frame pixels')
        origins=np.zeros(4)
    elif pixel_coordinates=='crop_local':
        if left_origin is None or right_origin is None:raise ValueError('Both actual crop origins required')
        origins=np.asarray([left_origin,right_origin],float)
        if origins.shape!=(2,2) or not np.isfinite(origins).all() or np.any(origins<0):raise ValueError('Invalid crop origins')
        origins=origins.ravel()
    else:raise ValueError('Unknown pixel coordinates')
    full=p+origins;finite=np.isfinite(full).all(axis=1)
    size=cal.get('config_snapshot',{}).get('full_frame_extraction',{}).get('sensor_size_px')
    if size:
        limits=np.tile(size,2);finite &= np.all((full>=0)&(full<limits),axis=1)
    xyz=np.full((len(p),3),np.nan);valid=np.zeros(len(p),bool);errors=np.full((len(p),2,2),np.nan)
    if finite.any():
        rec=reconstruct(full[finite,:2],full[finite,2:],cal['state'],cal['view_branches'])
        xyz[finite]=rec['X_mm'];valid[finite]=rec['valid_mask'];errors[finite]=rec['reprojection_errors_px']
    return dict(xyz_mm=xyz,valid_mask=valid,full_pixels=full,reprojection_errors_px=errors)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',required=True);p.add_argument('--matches',required=True)
    p.add_argument('--output',required=True,help='New .npz path')
    p.add_argument('--pixel-coordinates',choices=['full_frame','crop_local'],default='full_frame')
    p.add_argument('--left-origin',nargs=2,type=float);p.add_argument('--right-origin',nargs=2,type=float)
    a=p.parse_args();out=Path(a.output)
    if out.suffix.lower()!='.npz' or out.exists():p.error('Use a new .npz output path')
    cal=load_calibration(a.calibration)
    result=reconstruct_matches(np.load(a.matches,allow_pickle=False),cal,a.pixel_coordinates,a.left_origin,a.right_origin)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(out,**result,model_version=cal['model_version'])
    print('Saved',out,'valid points',int(result['valid_mask'].sum()),'of',len(result['valid_mask']))


if __name__=='__main__':main()
