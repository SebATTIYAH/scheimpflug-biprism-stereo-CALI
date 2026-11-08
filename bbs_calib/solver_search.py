"""Deterministic search accounting; stability ranges are not confidence intervals."""
import numpy as np
from scipy.optimize import least_squares


def options(profile='enhanced',starts=None,seed=None,refine_nfev=None):
    if profile not in ('enhanced','baseline'):raise ValueError('Unknown solver profile')
    n=(5 if profile=='enhanced' else 1) if starts is None else starts
    budget=150 if refine_nfev is None else refine_nfev
    if n<1 or budget<1:raise ValueError('--starts and --refine-nfev must be positive')
    if profile=='baseline' and (n!=1 or refine_nfev is not None):
        raise ValueError('baseline uses one original start without final refinement')
    value=73129 if seed is None else seed
    if value<0:raise ValueError('--search-seed must be nonnegative')
    return dict(profile=profile,starts=int(n),seed=int(value),refine_nfev=int(budget))


def robust_cost(residual,loss='linear',f_scale=1.):
    if not np.isfinite(f_scale) or f_scale<=0:raise ValueError('Invalid robust loss scale')
    z=(np.asarray(residual,float)/f_scale)**2
    if loss=='linear':rho=z
    elif loss=='soft_l1':rho=2*z/(np.sqrt(1+z)+1)
    elif loss=='huber':rho=np.where(z<=1,z,2*np.sqrt(z)-1)
    elif loss=='cauchy':rho=np.log1p(z)
    elif loss=='arctan':rho=np.arctan(z)
    else:raise ValueError('Unsupported loss: '+str(loss))
    return float(.5*f_scale*f_scale*np.sum(rho))


def refine(fun,x,low,high,loss='linear',f_scale=1.,max_nfev=150,valid=None,jac=None):
    """Accept dense central-difference TRF only if converged, valid and non-worsening."""
    before=robust_cost(fun(x),loss,f_scale)
    record=dict(method='sparse TRF/LSMR; grouped differences' if jac is not None else 'dense TRF; 3-point finite differences',before_cost=before,accepted=False)
    try:
        extra={} if jac is None else dict(tr_solver='lsmr',tr_options={'atol':1e-10,'btol':1e-10})
        fit=least_squares(fun,x,bounds=(low,high),method='trf',jac='3-point' if jac is None else jac,x_scale='jac',**extra,
            loss=loss,f_scale=f_scale,max_nfev=max_nfev,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        after=robust_cost(fun(fit.x),loss,f_scale)
        geometry_ok=True if valid is None else bool(valid(fit.x))
        accepted=bool(fit.success and geometry_ok and np.isfinite(after) and after<=before+1e-10*max(1.,abs(before)))
        record.update(after_cost=after,success=bool(fit.success),accepted=accepted,geometry_valid=geometry_ok,
            optimality=float(fit.optimality),nfev=int(fit.nfev),message=str(fit.message),status=int(fit.status))
        return (fit.x if accepted else x.copy()),record
    except (ValueError,np.linalg.LinAlgError) as exc:
        record.update(success=False,error=str(exc))
        return x.copy(),record


def candidate_key(record):
    # Invalid paths must not win on objective, and unconverged candidates are fallback only.
    score=record.get('score',float('inf'))
    return (not record.get('geometry_valid',False),not record.get('converged',False),score if np.isfinite(score) else float('inf'))


def global_values(K,c=None,omega=None,pixel=(3.69,3.69)):
    from bbs_calib.angles import effective_angles
    angles=effective_angles(K,pixel)
    values=dict(J=float(K[0,0]),K=float(K[1,1]),L=float(K[1,0]),
                **{k:angles[k] for k in ('alpha_deg','beta_deg','total_tilt_deg')})
    if c is not None:values.update(c_x_px=float(c[0]),c_y_px=float(c[1]))
    if omega is not None:values.update(zip(('omega_x_deg','omega_y_deg','omega_z_deg'),np.degrees(omega).tolist()))
    return values


def stability(records,selected_trial,seed):
    eligible=[r for r in records if r.get('geometry_valid') and r.get('converged') and np.isfinite(r.get('score',float('inf')))]
    best=min((r['score'] for r in eligible),default=None)
    tolerance=None if best is None else max(1e-8,.01*abs(best))
    near=[] if best is None else [r for r in eligible if r['score']<=best+tolerance]
    def ranges(items):
        if not items:return {}
        keys=set.intersection(*(set(r['values']) for r in items))
        return {k:dict(min=float(min(r['values'][k] for r in items)),max=float(max(r['values'][k] for r in items)),
                       spread=float(np.ptp([r['values'][k] for r in items]))) for k in sorted(keys)}
    near_ranges=ranges(near)
    sensitive=any(near_ranges.get(k,{}).get('spread',0)>.1 for k in ('alpha_deg','beta_deg','total_tilt_deg'))
    status='insufficient_converged_near_optimal_starts' if len(near)<2 else ('near_equal_cost_different_angles' if sensitive else 'agreement_at_tested_starts')
    return dict(selected_trial=selected_trial,search_seed=seed,attempted=len(records),eligible_count=len(eligible),
        near_optimal_count=len(near),near_optimal_trials=[r['trial'] for r in near],near_optimal_cost_tolerance=tolerance,
        all_eligible_ranges=ranges(eligible),near_optimal_ranges=near_ranges,status=status,
        selection='Valid geometry, then converged, then lowest training objective; validation not used for selection',
        angle_spread_warning_threshold_deg=.1,confidence_interval=False,global_uniqueness_proven=False,
        note='Ranges describe tested initializations only. 1% objective window and 0.1 degree warning are diagnostic conventions, not metrological uncertainty.',
        candidates=records)


def save_report(search,output):
    from .io import write_json,write_csv
    write_json(output/'optimizer_search.json',search)
    rows=[]
    for r in search['candidates']:
        rows.append(dict(trial=r['trial'],selected=r['trial']==search['selected_trial'],converged=r.get('converged',False),
            geometry_valid=r.get('geometry_valid',False),training_objective=r.get('score'),
            refinement_accepted=(r.get('refinement') or {}).get('accepted',False),**r.get('values',{}),failure=r.get('failed','')))
    write_csv(output/'optimizer_candidates.csv',rows)
    lines=['# Optimization stability report','',
        'Candidates are selected using the training objective only; held-out data do not affect selection. Reported ranges are not confidence intervals and do not establish global uniqueness or mechanical tilt-angle accuracy.','',
        'Status: `'+search['status']+'`; selected initialization: '+str(search['selected_trial'])+' (0-based).',
        'Attempts: '+str(search['attempted'])+'; valid and converged: '+str(search['eligible_count'])+'; near-optimal: '+str(search['near_optimal_count'])+'.','',
        '| Initialization | Converged | Valid ray path/projection | Training objective | Total tilt θ (°) |','|---|---|---|---:|---:|']
    for r in rows:
        lines.append('| %s | %s | %s | %s | %s |'%(r['trial'],r['converged'],r['geometry_valid'],r.get('training_objective'),r.get('total_tilt_deg')))
    lines+=['','The near-optimal window is 1% of the minimum training objective (absolute floor: 1e-8); an angular span over 0.1° triggers a warning. This is a diagnostic convention, not a measurement-accuracy threshold.',
        'Even when all initializations give similar results, assess parameter coupling, data coverage, and independent measurements before trusting the solution.',
        'If no valid candidate converges, the retained fallback is for troubleshooting only and does not constitute a successful calibration.']
    (output/'optimizer_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
