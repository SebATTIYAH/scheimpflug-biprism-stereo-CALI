from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .config import board_points, initial_state, validate_config, resolve_q_scale
from .io import read_csv
from .geometry import optical_system, path_validity, GeometryError, scheimpflug
from .initialization import initialize_pose
from .projection import object_points, project


def observations(path: Path, c: dict) -> dict:
    from .image_inputs import input_mode,crop_origins,origin_for
    origin_table=crop_origins(c)
    ids,P=board_points(c)
    pointmap=dict(zip(ids,P))
    records=read_csv(path)
    used,excluded,seen=[],[],set()
    for index,row in enumerate(records):
        key=(row.get('image_id'),row.get('view'),row.get('point_id'))
        if row.get('accepted','').lower() not in ('true','1') or row.get('review_status')!='confirmed':
            excluded.append(dict(row=index,reason='not accepted and confirmed',key=key))
            continue
        if key in seen:
            raise ValueError('Duplicate confirmed observation: '+str(key))
        seen.add(key)
        if key[1] not in ('left','right') or key[2] not in pointmap:
            raise ValueError('Unknown view or point ID: '+str(key))
        p=np.array([float(row['u_px']),float(row['v_px'])])
        if input_mode(c)=='paired_crops':
            space=row.get('coordinate_space')
            if space not in ('full_frame','crop_local'):
                raise ValueError('Cropped observations require explicit coordinate_space: full_frame or crop_local')
            origin=origin_for(c,key[0],key[1],origin_table)
            if space=='crop_local':
                if origin is None:
                    raise ValueError('Crop origin missing for '+str(key[:2])+': cannot use two unrelated local origins with shared K/c')
                p=p+origin
                row=dict(row,u_px=str(p[0]),v_px=str(p[1]),coordinate_space='full_frame',
                         crop_origin_x_px=str(origin[0]),crop_origin_y_px=str(origin[1]))
            elif origin is not None and row.get('source_u_px') and row.get('source_v_px'):
                restored=np.array([float(row['source_u_px']),float(row['source_v_px'])])+origin
                if not np.allclose(restored,p,atol=1e-5,rtol=0):
                    raise ValueError('Crop origin changed since detection; regenerate observations to avoid inconsistent coordinates')
        if not np.isfinite(p).all():
            raise ValueError('Nonfinite observed coordinates')
        if input_mode(c)=='full_frame' and c.get('full_frame_extraction',{}).get('detector')=='dual_grid_v1':
            if row.get('coordinate_space')!='full_frame':
                raise ValueError('Full-sensor dual-grid observations require coordinate_space=full_frame')
            source=np.array([float(row['source_u_px']),float(row['source_v_px'])])
            if not np.allclose(source,p,atol=1e-7,rtol=0):
                raise ValueError('Full-sensor observations must not contain crop offsets')
        W=np.eye(2)
        if c.get('optimization',{}).get('use_covariance',False):
            if row.get('covariance_status')!='validated':
                raise ValueError('Whitening requires supplied validated center covariance')
            cov=np.array([[float(row['cov_uu']),float(row['cov_uv'])],[float(row['cov_uv']),float(row['cov_vv'])]])
            W=np.linalg.solve(np.linalg.cholesky(cov),np.eye(2))
        used.append(dict(row=row,P=pointmap[key[2]],p=p,W=W,branch=c['views'][key[1]]['branch']))
    if not used:
        raise ValueError('No accepted, confirmed observations; review IDs before calibration')
    frames=sorted(set(x['row']['image_id'] for x in used))
    minimum=c.get('detection',{}).get('min_points_per_view',6)
    for frame in frames:
        sets=[{x['row']['point_id'] for x in used if x['row']['image_id']==frame and x['row']['view']==v} for v in ('left','right')]
        if min(map(len,sets))<minimum or len(sets[0]&sets[1])<minimum:
            raise ValueError('Frame '+frame+' needs >= '+str(minimum)+' confirmed paired points in both views')
    return dict(used=used,excluded=excluded,frames=frames)


def array_data(used: list, frames: list) -> dict:
    indices={f:i for i,f in enumerate(frames)}
    selected=[v for v in used if v['row']['image_id'] in indices]
    return dict(P=np.array([v['P'] for v in selected]),p=np.array([v['p'] for v in selected]),
                W=np.array([v['W'] for v in selected]),s=np.array([v['branch'] for v in selected]),
                frame=np.array([indices[v['row']['image_id']] for v in selected]),records=selected,frames=frames)


class Parameters:
    """Scaled variables: signed log magnitudes, shear/scale, rad/.1, translations/mm."""
    def __init__(self, state: dict, frames: list, c: dict):
        self.base=deepcopy(state)
        self.scale=np.sqrt(abs(state['K'][0,0]*state['K'][1,1]))
        self.sign=np.sign([state['K'][0,0],state['K'][1,1]])
        self.frames=frames
        self.c=c
        from .hardware_refinement import specifications
        self.hardware=specifications(c)
        self.n_global=8+len(self.hardware)
        self.names=['log_abs_j','log_abs_k','l_over_reference_scale','omega_x_over_0.1rad',
                    'omega_y_over_0.1rad','omega_z_over_0.1rad','kappa1','kappa2']
        self.names += ['prism.'+s['name']+'_normalized_delta' for s in self.hardware]
        self.names += [f+':'+a for f in frames for a in ('rx/0.1rad','ry/0.1rad','rz/0.1rad','tx_mm','ty_mm','tz_mm')]

    def pack(self,state:dict,poses:np.ndarray)->np.ndarray:
        K=state['K']
        return np.r_[np.log(abs(K[0,0])),np.log(abs(K[1,1])),K[1,0]/self.scale,
                     state['omega']/.1,state['kappa'],
                     [(state['prism'][s['name']]-s['nominal'])/s['max_delta'] for s in self.hardware],
                     (poses/np.array([.1,.1,.1,1,1,1])).ravel()]

    def unpack(self,x:np.ndarray)->tuple:
        state=deepcopy(self.base)
        state['K']=np.array([[self.sign[0]*np.exp(x[0]),0.],[x[2]*self.scale,self.sign[1]*np.exp(x[1])]])
        state['omega']=x[3:6]*.1
        state['kappa']=x[6:8].copy()
        for i,s in enumerate(self.hardware):
            state['prism'][s['name']]=s['nominal']+x[8+i]*s['max_delta']
        poses=x[self.n_global:].reshape(-1,6)*[.1,.1,.1,1,1,1]
        return state,poses

    def bounds(self,x:np.ndarray)->tuple:
        low=np.full(len(x),-np.inf); high=np.full(len(x),np.inf)
        settings=self.c.get('optimization',{})
        factor=settings.get('scale_factor_bounds',[.5,2.])
        low[:2]=x[:2]+np.log(factor[0]); high[:2]=x[:2]+np.log(factor[1])
        low[2],high[2]=-settings.get('max_shear_fraction',.5),settings.get('max_shear_fraction',.5)
        om=np.radians(settings.get('max_prism_rotvec_component_deg',12.))/.1
        low[3:6],high[3:6]=-om,om
        kd=settings.get('radial_bounds',[-.15,.15])
        low[6:8],high[6:8]=kd
        low[8:self.n_global],high[8:self.n_global]=-1.,1.
        for i in range(len(self.frames)):
            k=self.n_global+6*i
            low[k:k+3],high[k:k+3]=-np.pi/.1,np.pi/.1
            lim=settings.get('translation_abs_limit_mm',1000.)
            low[k+3:k+6],high[k+3:k+6]=-lim,lim
            low[k+5]=0.
        return low,high

    def active(self,stage:str)->np.ndarray:
        opt=self.c.get('optimization',{})
        global_indices=[]
        if stage!='A':
            global_indices += [i for i,v in enumerate(opt.get('release_K',[True]*3)) if v]
            global_indices += [3+i for i,v in enumerate(opt.get('release_omega',[True]*3)) if v]
        if stage=='C':
            global_indices += [6+i for i,v in enumerate(opt.get('release_radial',[True,True])) if v]
            global_indices += list(range(8,self.n_global))
        return np.array(global_indices+list(range(self.n_global,len(self.names))),int)


class Objective:
    def __init__(self,codec:Parameters,data:dict):
        from threading import Lock
        self._counter_lock=Lock()
        self.codec,self.data=codec,data
        self.invalid_calls=0
        self.invalid_reasons={}
        self.priors=[]
        for idx,name in enumerate(('x','y','z')):
            prior=codec.c.get('priors',{}).get('omega_'+name)
            if prior:
                if prior['sigma_deg']<=0:
                    raise ValueError('Prior sigma must be positive')
                self.priors.append((3+idx,np.radians(prior['mean_deg'])/.1,np.radians(prior['sigma_deg'])/.1))
        self.priors += [(8+i,0.,s['prior_sigma']/s['max_delta']) for i,s in enumerate(codec.hardware)]

    def raw(self,x:np.ndarray,penalty:bool=True)->np.ndarray:
        if self.codec.c.get('optimization',{}).get('resolved_backend')=='sparse':
            from .sparse_solver import vector_raw
            try:
                residual,invalid=vector_raw(self.codec,self.data,x,penalty)
                if not np.isfinite(residual).all():raise GeometryError('Nonfinite projection')
                # Only diagnostic counters are shared; projection arrays are local to each call.
                if invalid:
                    with self._counter_lock:
                        self.invalid_calls+=invalid
                        self.invalid_reasons['half/interface/aperture']=self.invalid_reasons.get('half/interface/aperture',0)+invalid
                return residual
            except GeometryError as exc:
                if not penalty:raise
                with self._counter_lock:
                    self.invalid_calls+=1
                    reason=str(exc);self.invalid_reasons[reason]=self.invalid_reasons.get(reason,0)+1
                return np.full_like(self.data['p'],1e5*(1+np.linalg.norm(x[3:6])**2))
        state,poses=self.codec.unpack(x)
        d=self.data
        pred=np.empty_like(d['p'])
        try:
            system=optical_system(state['prism'],state['omega'])
            for i in range(len(poses)):
                for s in ('L','R'):
                    mask=(d['frame']==i)&(d['s']==s)
                    X=object_points(d['P'][mask],poses[i])
                    pred[mask]=project(X,s,state,system)
                    if penalty:
                        valid,violation,_=path_validity(X,s,system,state['prism'])
                        if not valid.all():
                            self.invalid_calls+=1
                            self.invalid_reasons['half/interface/aperture']=self.invalid_reasons.get('half/interface/aperture',0)+1
                            # Fixed-length, positive smooth exterior penalty; never discard observations.
                            block=pred[mask]
                            block[~valid]=d['p'][mask][~valid]+(1000.+1000*violation[~valid,None])
                            pred[mask]=block
            residual=pred-d['p']
            if not np.isfinite(residual).all():
                raise GeometryError('Nonfinite projection')
            return residual
        except GeometryError as exc:
            if not penalty:
                raise
            self.invalid_calls+=1
            reason=str(exc)
            self.invalid_reasons[reason]=self.invalid_reasons.get(reason,0)+1
            return np.full_like(pred,1e5*(1+np.linalg.norm(x[3:6])**2))

    def __call__(self,x:np.ndarray)->np.ndarray:
        residual=np.einsum('nij,nj->ni',self.data['W'],self.raw(x)).ravel()
        return np.r_[residual,[(x[i]-mean)/sigma for i,mean,sigma in self.priors]]

    def sparsity(self,active:np.ndarray):
        n=len(self.data['p'])
        J=lil_matrix((2*n+len(self.priors),len(self.codec.names)),dtype=int)
        ng=self.codec.n_global
        J[:2*n,:ng]=1
        for i,frame in enumerate(self.data['frame']):
            J[2*i:2*i+2,ng+6*frame:ng+6*frame+6]=1
        for row,(idx,_,_) in enumerate(self.priors):
            J[2*n+row,idx]=1
        return J[:,active].tocsr()


def finite_jacobian(function,x:np.ndarray,active:np.ndarray)->np.ndarray:
    cols=[]
    for i in active:
        step=2e-5*max(1.,abs(x[i]))
        p,m=x.copy(),x.copy(); p[i]+=step; m[i]-=step
        cols.append((function(p)-function(m))/(2*step))
    return np.column_stack(cols)


def matrix_diagnostics(J:np.ndarray,names:list,relative:float=1e-6)->dict:
    _,s,Vt=np.linalg.svd(J,full_matrices=False)
    threshold=s[0]*relative if len(s) else 0.
    rank=int(np.sum(s>threshold))
    norms=np.linalg.norm(J,axis=0)
    corr=(J.T@J)/np.maximum(np.outer(norms,norms),1e-30)
    # Formal unit-noise local parameter correlations; never present as measured precision.
    parameter_correlation=None
    if rank==J.shape[1]:
        covariance=(Vt.T/(s*s))@Vt
        sd=np.sqrt(np.maximum(np.diag(covariance),0))
        parameter_correlation=covariance/np.maximum(np.outer(sd,sd),1e-30)
    pairs=[]
    for i in range(len(names)):
        for j in range(i):
            if np.isfinite(corr[i,j]) and abs(corr[i,j])>.98:
                pairs.append(dict(a=names[i],b=names[j],cosine_correlation=float(corr[i,j])))
    return dict(rank=rank,columns=J.shape[1],singular_values=s,rank_relative_threshold=relative,
                scaled_parameter_names=names,column_norms=norms,
                near_zero_columns=[names[i] for i,v in enumerate(norms) if v<max(norms.max(),1.)*relative],
                high_column_correlations=pairs,local_parameter_correlation=parameter_correlation,
                weakest_right_singular_vector=dict(zip(names,Vt[-1].tolist())) if len(s) else {},
                full_column_rank=rank==J.shape[1],parameter_standard_deviations=None,
                uncertainty_note='No metric covariance asserted: systematic errors and model adequacy not established')


def calibrate(c:dict,obs:dict)->dict:
    validate_config(c,'calibrate')
    start_time=time.perf_counter()
    frames=obs['frames']
    if len(frames)<c.get('optimization',{}).get('min_training_frames',4):
        raise ValueError('Insufficient distinct frames for joint calibration')
    rng=np.random.default_rng(c.get('seed',1234))
    valid_frames=c.get('validation',{}).get('frames')
    if valid_frames is None:
        count=int(round(len(frames)*c.get('validation',{}).get('fraction',.25)))
        valid_frames=sorted(rng.choice(frames,count,replace=False).tolist()) if count else []
    if not set(valid_frames)<=set(frames):
        raise ValueError('Configured validation frame not found in observations')
    training=[f for f in frames if f not in valid_frames]
    if len(training)<c.get('optimization',{}).get('min_training_frames',4):
        raise ValueError('Need >=4 training frames after whole-frame validation split')
    data=array_data(obs['used'],training)
    normalization=resolve_q_scale(c,data['p'],training)
    print('[solver] q_scale_mm=%.9g (%s); fixed for fit and reconstruction' %
          (normalization['q_scale_mm'],normalization['source']),flush=True)
    opt=c.get('optimization',{})
    backend=opt.get('backend','auto')
    if backend not in ('auto','legacy','sparse'):raise ValueError('Unknown backend')
    workers=opt.get('workers',1)
    if not isinstance(workers,int) or not 1<=workers<=32:raise ValueError('workers must be an integer from 1 to 32')
    if backend=='legacy' and workers>1:raise ValueError('workers > 1 requires auto/sparse backend')
    fast=backend=='sparse' or (backend=='auto' and (len(training)>50 or workers>1))
    opt['resolved_backend']='sparse' if fast else 'legacy'
    from .hardware_refinement import specifications
    hardware=specifications(c)
    print('[solver] backend=%s, workers=%d, training frames=%d, parameters=%d'%(opt['resolved_backend'],workers,len(training),8+len(hardware)+6*len(training)),flush=True)
    if hardware:
        print('[solver] bounded hardware refinement: '+', '.join(s['name'] for s in hardware)+'; nominal priors are not independent measurements',flush=True)
    enhanced=opt.get('solver_profile')=='enhanced'
    from .solver_search import robust_cost,refine,candidate_key,global_values,stability
    search_seed=opt.get('search_seed',73129)
    search_rng=np.random.default_rng(search_seed)
    search_records=[]
    attempts=[]; best=None
    base=initial_state(c)
    for trial in range(opt.get('multistart',2)):
        state=deepcopy(base)
        if enhanced:
            print('[solver] start %d/%d'%(trial+1,opt.get('multistart',2)),flush=True)
        if trial and enhanced:
            release_K=opt.get('release_K',[True]*3)
            for axis in (0,1):
                if release_K[axis]:state['K'][axis,axis]*=np.exp(search_rng.normal(0,.01))
            if release_K[2]:state['K'][1,0]+=np.sqrt(abs(base['K'][0,0]*base['K'][1,1]))*search_rng.normal(0,.002)
            state['omega']+=np.array(opt.get('release_omega',[True]*3))*search_rng.normal(0,np.radians(.5),3)
        elif trial:
            release=np.array(opt.get('release_omega',[True]*3))
            state['omega']+=release*rng.normal(0,np.radians(opt.get('multistart_omega_sigma_deg',.3)),3)
        try:
            poses=[]; initialization=[]
            for i,f in enumerate(training):
                mask=data['frame']==i
                pose,diag=initialize_pose(data['P'][mask],data['p'][mask],data['s'][mask],state)
                X=object_points(data['P'][mask],pose)
                if fast:
                    residual=np.empty_like(data['p'][mask]);branches=data['s'][mask];pixels=data['p'][mask]
                    for branch in ('L','R'):
                        selected=branches==branch;residual[selected]=project(X[selected],branch,state)-pixels[selected]
                    if (i+1)%50==0:print('[solver] initialized %d/%d poses'%(i+1,len(training)),flush=True)
                else:
                    residual=[]
                    for point,pixel,branch in zip(X,data['p'][mask],data['s'][mask]):
                        residual.append(project(point[None,:],branch,state)[0]-pixel)
                initial_rms=float(np.sqrt(np.mean(np.sum(np.array(residual)**2,axis=1))))
                diag['joint_branch_initial_rms_px']=initial_rms
                if initial_rms>opt.get('max_initial_pose_rms_px',15.):
                    raise ValueError('Frame '+f+' fails joint-branch pose consistency: check IDs, branch mapping or initial hardware parameters')
                poses.append(pose); initialization.append(dict(image_id=f,**diag))
            codec=Parameters(base if enhanced else state,training,c)
            x=codec.pack(state,np.array(poses))
            lower,upper=codec.bounds(codec.pack(base,np.array(poses)) if enhanced else x)
            if np.any(x<=lower) or np.any(x>=upper):
                raise ValueError('Initialization outside configured bounds; inspect units/branch mapping/priors')
            objective=Objective(codec,data)
            stages=[]
            for stage in ('A','B','C'):
                if stage=='C' and not any(opt.get('release_radial',[True,True])) and not codec.hardware:
                    continue
                active=codec.active(stage)
                frozen=x.copy()
                def fun(y:np.ndarray)->np.ndarray:
                    full=frozen.copy(); full[active]=y
                    return objective(full)
                print('[solver] start %d stage %s'%(trial+1,stage),flush=True)
                grouped=None
                kwargs=dict(jac_sparsity=objective.sparsity(active))
                if fast:
                    from .sparse_solver import GroupedJacobian
                    grouped=GroupedJacobian(fun,objective.sparsity(active),active,workers,lower[active],upper[active],label='start %d stage %s'%(trial+1,stage),n_global=codec.n_global)
                    kwargs=dict(jac=grouped,tr_solver='lsmr')
                try:
                    result=least_squares(fun,x[active],method='trf',bounds=(lower[active],upper[active]),
                                     loss=opt.get('loss','soft_l1'),f_scale=opt.get('f_scale_px',.5),
                                     x_scale='jac',**kwargs,
                                     ftol=opt.get('tolerance',1e-8),xtol=opt.get('tolerance',1e-8),gtol=opt.get('tolerance',1e-8),
                                     tr_options={'atol':1e-10,'btol':1e-10},
                                     max_nfev=opt.get('max_nfev',180))
                finally:
                    if grouped:grouped.close()
                print('[solver] stage %s done: nfev=%d, cost=%.9g, converged=%s'%(stage,result.nfev,result.cost,result.success),flush=True)
                x[active]=result.x
                stages.append(dict(stage=stage,success=bool(result.success),status=int(result.status),message=result.message,
                                   nfev=result.nfev,robust_objective=float(result.cost),optimality=float(result.optimality),
                                   active_parameters=[codec.names[i] for i in active]))
            polish=None
            if enhanced:
                frozen=x.copy()
                def expanded(y):
                    full=frozen.copy();full[active]=y;return full
                def valid_candidate(y):
                    st,ps=codec.unpack(expanded(y));system=optical_system(st['prism'],st['omega'])
                    return all(path_validity(object_points(data['P'][(data['frame']==i)&(data['s']==s)],ps[i]),s,system,st['prism'])[0].all()
                               for i in range(len(ps)) for s in ('L','R'))
                print('[solver] %s refinement for start %d'%('sparse' if fast else 'dense',trial+1),flush=True)
                grouped=None;polish_fun=lambda y:objective(expanded(y))
                if fast:grouped=GroupedJacobian(polish_fun,objective.sparsity(active),active,workers,lower[active],upper[active],label='start %d polish'%(trial+1),n_global=codec.n_global)
                try:
                    y,polish=refine(polish_fun,x[active],lower[active],upper[active],
                        loss=opt.get('loss','soft_l1'),f_scale=opt.get('f_scale_px',.5),max_nfev=opt.get('refine_nfev',150),valid=valid_candidate,jac=grouped)
                finally:
                    if grouped:grouped.close()
                x[active]=y
                if polish['accepted']:
                    stages.append(dict(stage='polish',success=True,status=polish['status'],message=polish['message'],nfev=polish['nfev'],
                        robust_objective=polish['after_cost'],optimality=polish['optimality'],active_parameters=[codec.names[i] for i in active]))
            raw=objective.raw(x,penalty=False)
            total=float(np.sum(objective(x)**2))
            selection_score=robust_cost(objective(x),opt.get('loss','soft_l1'),opt.get('f_scale_px',.5)) if enhanced else total
            fitted,trial_poses=codec.unpack(x);system=optical_system(fitted['prism'],fitted['omega'])
            geometry_ok=all(path_validity(object_points(data['P'][(data['frame']==i)&(data['s']==s)],trial_poses[i]),s,system,fitted['prism'])[0].all()
                            for i in range(len(trial_poses)) for s in ('L','R'))
            search_record=dict(trial=trial,score=selection_score,geometry_valid=bool(geometry_ok),converged=bool(stages[-1]['success']),
                at_bounds=[codec.names[i] for i in active if min(x[i]-lower[i],upper[i]-x[i])<1e-4],
                values=global_values(fitted['K'],fitted['c'],fitted['omega'],c['camera'].get('pixel_size_um') or (1,1)),
                initial_K=state['K'].tolist(),initial_omega_deg=np.degrees(state['omega']).tolist(),refinement=polish,
                kappa=fitted['kappa'].tolist(),stages=stages)
            search_record['values'].update({'prism.'+s['name']:float(fitted['prism'][s['name']]) for s in codec.hardware})
            search_records.append(search_record)
            attempts.append(dict(trial=trial,objective_unrobust_with_priors=total,stages=stages,
                                 global_scaled_solution=x[:codec.n_global].copy(),invalid_objective_calls=objective.invalid_calls))
            key=candidate_key(search_record) if enhanced else (total,)
            if best is None or key<best['selection_key']:
                best=dict(score=selection_score,selection_key=key,trial=trial,x=x,codec=codec,objective=objective,active=active,raw=raw,stages=stages,
                          lower=lower,upper=upper,initialization=initialization)
        except (ValueError,np.linalg.LinAlgError) as exc:
            attempts.append(dict(trial=trial,failed=str(exc)))
            search_records.append(dict(trial=trial,failed=str(exc),converged=False,geometry_valid=False))
    if best is None:
        raise ValueError('All initializations failed: '+str(attempts))
    x,codec,objective,active=best['x'],best['codec'],best['objective'],best['active']
    state,poses=codec.unpack(x)
    # Unweighted, unrobust DATA Jacobian excludes priors. Scaled optimization variables are explicit.
    print('[solver] diagnostics: '+('pose-block sparse' if fast else 'full dense'),flush=True)
    if fast:
        from .sparse_solver import diagnostics as sparse_diagnostics
        data_diag,augmented_diag,global_diag=sparse_diagnostics(objective,x,active)
    else:
        data_diag,augmented_diag,global_diag=dense_diagnostics(objective,x,active,codec)
    return finish_calibration(c,obs,start_time,valid_frames,training,data,opt,enhanced,search_seed,search_records,attempts,best,x,codec,objective,active,state,poses,data_diag,augmented_diag,global_diag)


def dense_diagnostics(objective,x,active,codec):
    J=finite_jacobian(lambda v:objective.raw(v,False).ravel(),x,active)
    data_diag=matrix_diagnostics(J,[codec.names[i] for i in active])
    Ja=finite_jacobian(objective,x,active)
    augmented_diag=matrix_diagnostics(Ja,[codec.names[i] for i in active])
    ng=int(np.sum(active<codec.n_global))
    if ng:
        Jg,Jp=J[:,:ng],J[:,ng:]
        U,s,_=np.linalg.svd(Jp,full_matrices=False)
        U=U[:,s>s[0]*1e-9]
        reduced=Jg-U@(U.T@Jg)
        global_diag=matrix_diagnostics(reduced,[codec.names[i] for i in active[:ng]])
    else:
        global_diag=dict(full_column_rank=True,rank=0,columns=0,singular_values=[])
    return data_diag,augmented_diag,global_diag


def finish_calibration(c,obs,start_time,valid_frames,training,data,opt,enhanced,search_seed,search_records,attempts,best,x,codec,objective,active,state,poses,data_diag,augmented_diag,global_diag):
    from .solver_search import stability
    system=optical_system(state['prism'],state['omega'])
    valid=[]
    for i,pose in enumerate(poses):
        for s in ('L','R'):
            mask=(data['frame']==i)&(data['s']==s)
            good,_,_=path_validity(object_points(data['P'][mask],pose),s,system,state['prism'])
            valid.extend(good.tolist())
    warnings=[]
    identifiable=data_diag['full_column_rank'] and global_diag['full_column_rank']
    if not identifiable:
        warnings.append('Data do not identify all released parameters. Convergence/prior regularization does not remove this ambiguity.')
    if global_diag.get('high_column_correlations'):
        warnings.append('Strong global parameter sensitivity correlations; full numerical rank is not mechanical precision evidence.')
    if not state['prism'].get('aperture_mm'):
        warnings.append('Only infinite planes, half-faces and interface ordering validated; actual aperture unknown.')
    if not valid_frames:
        warnings.append('No independent whole-frame validation set.')
    if objective.priors:
        warnings.append('Priors used. Data-only and augmented Jacobians are reported separately.')
    if not all(s['success'] for s in best['stages']):
        warnings.append('At least one optimization stage did not meet convergence criteria.')
    near=[codec.names[i] for i in active if min(x[i]-best['lower'][i],best['upper'][i]-x[i])<1e-4]
    if near:
        warnings.append('Parameters at bounds: '+', '.join(near))
    solutions=[a['global_scaled_solution'] for a in attempts if 'global_scaled_solution' in a]
    diagnostics=dict(data_jacobian=data_diag,augmented_weighted_jacobian=augmented_diag,
                     execution=dict(backend=opt.get('resolved_backend'),workers=opt.get('workers',1)),
                     global_after_eliminating_poses=global_diag,at_bounds=near,multistart=attempts,
                     multistart_scaled_range=np.ptp(solutions,axis=0) if len(solutions)>1 else None,
                     invalid_evaluation_reasons=objective.invalid_reasons,initialization=best['initialization'],
                     warnings=warnings,excluded_observations=obs['excluded'])
    if enhanced:
        diagnostics['optimizer_search']=stability(search_records,best['trial'],search_seed)
        diagnostics['optimizer_search'].update(loss=opt.get('loss','soft_l1'),f_scale=opt.get('f_scale_px',.5),
            common_parameter_bounds=True,heldout_used_for_selection=False,
            perturbations=dict(log_abs_J_K_sigma=.01,L_over_reference_sigma=.002,prism_rotation_sigma_deg=.5),
            refine_nfev=opt.get('refine_nfev',150))
        if diagnostics['optimizer_search']['status']!='agreement_at_tested_starts':
            warnings.append('Initial-value search: '+diagnostics['optimizer_search']['status'])
    from .hardware_refinement import report as hardware_report
    hardware=hardware_report(c,state,identifiable)
    if hardware:
        diagnostics['hardware_refinement']=hardware
        warnings.append(hardware['note'])
    angles=derive_angles(state,c,identifiable and not near)
    status=dict(implementation_tested=False,optimization_converged=bool(best['stages'][-1]['success']),
                geometry_valid=bool(all(valid)),identifiability_passed=bool(identifiable),external_validation_available=False)
    return dict(state=state,poses=dict(zip(training,poses)),training_frames=training,validation_frames=valid_frames,
                diagnostics=diagnostics,status=status,mechanical_angles=angles,data=data,raw_errors=best['raw'],
                whitened_sum_squares=float(np.sum(np.einsum('nij,nj->ni',data['W'],best['raw'])**2)),
                robust_objective=best['stages'][-1]['robust_objective'],elapsed_seconds=time.perf_counter()-start_time)


def derive_angles(state:dict,c:dict,identifiable:bool)->dict:
    result=dict(alpha_deg=None,beta_deg=None,m_px_per_mm=None,status='not_identifiable',reasons=[])
    cam=c['camera']; opt=c.get('optimization',{})
    from .hardware_refinement import specifications
    if specifications(c):
        result['reasons'].append('Hardware refined with nominal priors, not independently measured; mechanical certification withheld')
    if opt.get('mode','effective')!='physical_constrained':
        result['reasons'].append('Effective mode: mechanical interpretation not requested')
    if not identifiable:
        result['reasons'].append('Data identifiability/boundary checks failed')
    pixel=cam.get('pixel_size_um')
    if not pixel or min(pixel)<=0:
        result['reasons'].append('Measured pixel aspect ratio is missing')
    if any(opt.get('release_omega',[True]*3)) or cam.get('omega_source')!='measured_prior':
        result['reasons'].append('First-version physical interpretation requires independently measured fixed prism orientation')
    sign=cam.get('alpha_sign')
    if sign not in (-1,1):
        result['reasons'].append('Independent tilt sign/assembly convention is missing')
    if result['reasons']:
        return result
    K=state['K'].copy()
    K[1]*=pixel[1]/pixel[0]
    j,k,l=K[0,0],K[1,1],K[1,0]
    roots=np.roots([(k/j)**2,-((l/j)**2+(k/j)**2+1),1.])
    candidates=[]
    for root in roots:
        if abs(root.imag)>1e-8 or not 0<root.real<=1+1e-10:
            continue
        alpha=sign*np.arccos(np.sqrt(min(root.real,1)))
        m=k*np.cos(alpha)
        cosb=m/j
        if not 0<cosb<=1+1e-10:
            continue
        beta=np.arccos(min(cosb,1))
        if abs(alpha)<1e-5 or abs(beta)<1e-5:
            continue
        beta*=np.sign(-l/(m*np.tan(alpha)))
        if np.allclose(m*scheimpflug(alpha,beta),K,rtol=1e-6,atol=1e-6):
            candidate=[float(np.degrees(alpha)),float(np.degrees(beta)),float(m)]
            if not any(np.allclose(candidate,v) for v in candidates):
                candidates.append(candidate)
    if len(candidates)!=1:
        result['reasons'].append('No unique physical root, or near-zero-angle sensitivity is too weak')
        return result
    result.update(zip(('alpha_deg','beta_deg','m_px_per_mm'),candidates[0]))
    result['status']='derived_under_independent_constraints'
    result['opposite_sign_branch_deg']=[-candidates[0][0],-candidates[0][1]]
    return result
