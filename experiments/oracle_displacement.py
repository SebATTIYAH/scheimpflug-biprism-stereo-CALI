"""Fixed-calibration numerical checks; NOT a particle-image velocimetry test.

Noise is independent Gaussian error on each raw pixel coordinate at each of
two instants. Correspondences and optical calibration are supplied exactly.
The reference inverse uses the same q-space least-squares convention as the
production inverse; predictions use its analytic differential, not Monte Carlo.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation
from bbs_calib.geometry import optical_system, scheimpflug, path_validity
from bbs_calib.projection import project
from bbs_calib.reconstruction import reconstruct_points


def scene(wedge=15.):
    return dict(K=-100*scheimpflug(*np.radians([8.,5.])), c=np.array([500.,250.]),
                omega=np.radians([.7,-.5,.8]), kappa=np.array([.004,-.001]), q_scale=6.,
                prism=dict(refractive_index=1.5, wedge_L_deg=wedge,
                           wedge_R_deg=wedge+.2, thickness_mm=2.,
                           aperture_mm=dict(x=[-7.,7.],y=[-4.,4.])))


def independent_trace(X, prism, omega, branch):
    """Scalar intersections and angle Snell law; no production ray matrices."""
    R=Rotation.from_rotvec(omega).as_matrix()
    incoming=R.T@np.array([0.,0.,1.])
    d2=np.r_[incoming[:2]/prism['refractive_index'], 0.]
    d2[2]=np.sqrt(1.-d2@d2)
    tau=(-1. if branch=='L' else 1.)*np.tan(np.radians(prism['wedge_'+branch+'_deg']))
    normal=np.array([tau,0.,1.]); normal/=np.linalg.norm(normal)
    theta=np.arccos(np.clip(d2@normal,-1.,1.))
    tangent=d2-(d2@normal)*normal
    if np.linalg.norm(tangent)<1e-14:
        d1=normal
    else:
        tangent/=np.linalg.norm(tangent)
        theta_out=np.arcsin(prism['refractive_index']*np.sin(theta))
        d1=np.cos(theta_out)*normal+np.sin(theta_out)*tangent
    front=X+((prism['thickness_mm']-tau*X[0]-X[2])/(tau*d1[0]+d1[2]))*d1
    back=front-front[2]/d2[2]*d2
    return (R@back)[:2]


def inverse_jacobian(X,state,system):
    """Derivative of the implemented q-space reconstruction with respect to p."""
    blocks=[]
    for branch in ('R','L'):  # Image left -> physical R, image right -> physical L.
        q=system[branch]['M']@X+system[branch]['b']
        k1,k2=state['kappa']; s2=state['q_scale']**2; r2=q@q/s2
        derivative=(1+k1*r2+k2*r2*r2)*np.eye(2)+2*(k1+2*k2*r2)/s2*np.outer(q,q)
        blocks.append(np.linalg.solve(derivative,np.linalg.inv(state['K'])))
    H=np.vstack([system[b]['M'] for b in ('R','L')])
    return np.linalg.pinv(H)@block_diag(*blocks)


def run(output, seed=314159, samples=5000):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new or empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    checks=[]; rows=[]
    X=np.array([.1,.15,20.]); displacement=np.array([.025,-.015,.04])
    # The same two independently drawn noise arrays are scaled between sigma
    # levels within a geometry. Rows therefore must not be treated independent.
    for wedge in (5.,10.,15.):
        state=scene(wedge); system=optical_system(state['prism'],state['omega'])
        calibration=dict(state=state,view_branches=dict(left='R',right='L'),radius_limit_mm=10.)
        base=np.array([X,X+displacement])
        pixels=np.column_stack([project(base,b,state,system) for b in ('R','L')])
        H=np.vstack([system[b]['M'] for b in ('R','L')])
        ray_error=[]; affine_error=[]
        for _ in range(100):
            point=rng.uniform([-.5,-.5,18.],[.5,.5,22.])
            for branch in ('R','L'):
                ref=independent_trace(point,state['prism'],state['omega'],branch)
                ray_error.append(np.linalg.norm(system[branch]['M']@point+system[branch]['b']-ref))
                ideal=state['K']@(system[branch]['M']@point+system[branch]['b'])+state['c']
                zero=dict(state,kappa=np.zeros(2))
                affine_error.append(np.linalg.norm(project(point[None,:],branch,zero,system)[0]-ideal))
        noiseless=reconstruct_points(pixels[:,:2],pixels[:,2:],calibration)
        if not noiseless['geometry_valid']:
            raise RuntimeError('Invalid synthetic ray geometry')
        cov_unit=sum(J@J.T for J in [inverse_jacobian(p,state,system) for p in base])
        checks.append(dict(left_wedge_deg=wedge,right_wedge_deg=wedge+.2,
                           ray_check_count=len(ray_error),max_ray_difference_mm=max(ray_error),
                           max_affine_difference_px=max(affine_error),
                           noiseless_max_error_mm=float(np.max(np.abs(noiseless['X_mm']-base))),
                           ray_matrix_condition=float(np.linalg.cond(H)),
                           displacement_covariance_per_sigma2_mm2=cov_unit.tolist()))
        noise=rng.normal(size=(2,samples,4))
        for sigma in (0.01,0.05,0.1,0.2):
            recovered=[]; all_valid=True
            for time in (0,1):
                noisy=pixels[time]+sigma*noise[time]
                result=reconstruct_points(noisy[:,:2],noisy[:,2:],calibration)
                all_valid &= bool(result['geometry_valid'])
                recovered.append(result['X_mm'])
            if not all_valid:
                raise RuntimeError('Noise moved samples outside tested geometry')
            error=recovered[1]-recovered[0]-displacement
            predicted=sigma*np.sqrt(np.diag(cov_unit))
            observed=np.std(error,axis=0,ddof=1)
            row=dict(left_wedge_deg=wedge,right_wedge_deg=wedge+.2,sigma_px=sigma,
                     samples=samples,valid_count=samples,
                     vector_rmse_mm=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))))
            for i,axis in enumerate('xyz'):
                row[axis+'_predicted_std_mm']=float(predicted[i])
                row[axis+'_observed_std_mm']=float(observed[i])
                row[axis+'_bias_mm']=float(error[:,i].mean())
                row[axis+'_relative_std_error']=float(observed[i]/predicted[i]-1)
            rows.append(row)
    with (output/'displacement_noise.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(kind='oracle_calibration_coordinate_noise_check',seed=seed,samples_per_condition=samples,
                point_mm=X.tolist(),displacement_mm=displacement.tolist(),geometry_checks=checks,
                noise_assumptions='Independent zero-mean Gaussian pixel components across views and two instants; reused standardized draws across sigma levels within one geometry.',
                excluded_effects=['calibration uncertainty','image formation','particle detection and correlation','mismatching','spatially correlated errors','flow dynamics'],
                velocity_note='Divide displacement by a measured time interval; no time interval or measured velocity is asserted here.',
                environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.system()),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'conditions':len(rows),'samples_per_condition':samples,'max_ray_difference_mm':max(v['max_ray_difference_mm'] for v in checks),
                      'max_relative_std_difference':max(abs(r[a+'_relative_std_error']) for r in rows for a in 'xyz')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=314159)
    parser.add_argument('--samples',type=int,default=5000)
    args=parser.parse_args()
    if args.samples<2: parser.error('--samples must be at least 2')
    run(args.output,args.seed,args.samples)
