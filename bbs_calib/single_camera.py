"""Hu et al. 2022, Eqs. 13-29: single bi-telecentric Scheimpflug calibration.

No biprism. Planar pose has only five observable parameters; tz is absent.
The equivalent +/- depth-normal pose branch does not determine tilt signs.
"""
from copy import deepcopy
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from .optimization import matrix_diagnostics,finite_jacobian

def affine(P,p):
    return np.linalg.lstsq(np.column_stack([P[:,:2],np.ones(len(P))]),p,rcond=None)[0].T

def intrinsic_init(frames):
    H=[affine(f['P'],f['p']) for f in frames]
    scale=float(np.median([np.linalg.svd(h[:,:2],compute_uv=False)[0] for h in H]))
    design=[];target=[]
    for h in H:
        a,b=h[0,:2]/scale;d,e=h[1,:2]/scale
        design.append([1,-(a*a+b*b),-(d*d+e*e),2*(a*d+b*e)])
        target.append(-(a*e-b*d)**2)
    D=np.array(design);y=np.array(target);S,_,rank,sv=np.linalg.lstsq(D,y,rcond=None)
    if rank<4:raise ValueError('Hu Eq.20 rank below 4: insufficient independent board tilts')
    if S[2]<=0 or S[0]/S[2]<=0:raise ValueError('Hu Eq.21 nonphysical linear solution; inspect pose diversity/points')
    j=-np.sqrt(S[2]);k=-np.sqrt(S[0]/S[2]);l=-S[3]/np.sqrt(S[2])
    def fun(x):
        j,k=-np.exp(x[:2]);l=x[2]
        return np.array([j*j*k*k,k*k+l*l,j*j,j*l])-S
    fit=least_squares(fun,[np.log(-j),np.log(-k),l],method='lm',max_nfev=1000)
    j,k=-np.exp(fit.x[:2]);l=fit.x[2]
    K=scale*np.array([[j,0],[l,k]])
    return K,dict(equation='Hu Eq.20/21/22',scale_px_per_mm=scale,S_normalized=S,
                  design_rank=int(rank),design_singular_values=sv,design_condition=float(sv[0]/sv[-1]),
                  homographies=H,initial_K=K)

def pose_init(frame,K,c):
    H=affine(frame['P'],frame['p']);M=np.linalg.solve(K,H[:,:2])
    U,s,Vt=np.linalg.svd(M)
    # Complete the 2x2 orthographic block with a proper rotation (one depth-sign branch).
    U3=np.eye(3);U3[:2,:2]=U;U3[2,2]=np.linalg.det(U)
    V3=np.eye(3);V3[:2,:2]=Vt;V3[2,2]=np.linalg.det(Vt)
    R=U3@Rotation.from_rotvec([np.arccos(np.clip(s[1],0,1)),0,0]).as_matrix()@V3
    return np.r_[Rotation.from_matrix(R).as_rotvec(),np.linalg.solve(K,H[:,2]-c)]

def projection(P,pose,K,c,kappa,scale):
    q=(P@Rotation.from_rotvec(pose[:3]).as_matrix().T)[:,:2]+pose[3:5]
    r2=np.sum(q*q,axis=1)/scale**2
    distorted=q*(1+kappa[0]*r2+kappa[1]*r2**2+kappa[2]*r2**3)[:,None]
    return distorted@K.T+c

def fit_pose(frame,K,c,kappa,scale,mask=None):
    sub=frame if mask is None else dict(P=frame['P'][mask],p=frame['p'][mask])
    init=pose_init(sub,K,c)
    return least_squares(lambda pose:(projection(sub['P'],pose,K,c,kappa,scale)-sub['p']).ravel(),
                         init,method='lm',max_nfev=500).x

class SingleObjective:
    def __init__(self,frames,K,c,scale=60):
        self.frames=frames;self.c=np.array(c,float);self.scale=scale
        self.reference=np.sqrt(abs(K[0,0]*K[1,1]))
        self.names=['log_abs_j','log_abs_k','l_over_reference','c_x/100px','c_y/100px','radial_r2','radial_r4','radial_r6']
        self.names += [f['id']+':'+a for f in frames for a in ['rx/.1rad','ry/.1rad','rz/.1rad','tx_mm','ty_mm']]
    def pack(self,K,c,kappa,poses):
        return np.r_[np.log(abs(K[0,0])),np.log(abs(K[1,1])),K[1,0]/self.reference,
                     (c-self.c)/100,kappa,(poses/[.1,.1,.1,1,1]).ravel()]
    def unpack(self,x):
        K=np.array([[-np.exp(x[0]),0],[x[2]*self.reference,-np.exp(x[1])]])
        return K,self.c+100*x[3:5],x[5:8],x[8:].reshape(-1,5)*[.1,.1,.1,1,1]
    def __call__(self,x):
        K,c,kappa,poses=self.unpack(x)
        return np.concatenate([(projection(f['P'],pose,K,c,kappa,self.scale)-f['p']).ravel() for f,pose in zip(self.frames,poses)])
    def sparsity(self,active):
        n=sum(2*len(f['P']) for f in self.frames);J=lil_matrix((n,len(self.names)),dtype=int);J[:,:8]=1
        start=0
        for i,f in enumerate(self.frames):
            end=start+2*len(f['P']);J[start:end,8+5*i:13+5*i]=1;start=end
        return J[:,active].tocsr()

def calibrate_single(frames,shape,release_center=True,zero_tilt=False,diagnostics=True,initial_override=None,polish_nfev=None):
    K,init=intrinsic_init(frames);c=(np.array(shape,float)-1)/2;scale=60.
    base_K=K.copy();base_c=c.copy()
    if initial_override is not None:
        K=np.asarray(initial_override['K'],float).copy()
        if release_center:c=np.asarray(initial_override['c'],float).copy()
    poses=np.array([fit_pose(f,K,c,np.zeros(3),scale) for f in frames])
    obj=SingleObjective(frames,base_K,base_c,scale);x=obj.pack(K,c,np.zeros(3),poses)
    anchor=obj.pack(base_K,base_c,np.zeros(3),poses)
    low=np.full(len(x),-np.inf);high=np.full(len(x),np.inf)
    low[:2]=anchor[:2]+np.log(.5);high[:2]=anchor[:2]+np.log(2.)
    low[2],high[2]=-.5,.5
    low[3:5]=-np.array(shape)/200;high[3:5]=np.array(shape)/200
    low[5:8],high[5:8]=-1,1
    if zero_tilt:x[0]=np.mean(x[:2]);x[1]=x[0];x[2]=0
    stages=[];solutions=[]
    for stage in ('geometry','radial','joint'):
        globals_=([0] if zero_tilt else [0,1,2])
        if stage=='radial':active=np.array([5,6,7])
        else:active=np.array(globals_+([3,4] if stage=='joint' and release_center else [])+([5,6,7] if stage=='joint' else [])+list(range(8,len(x))))
        frozen=x.copy()
        def expand(y):
            full=frozen.copy();full[active]=y
            if zero_tilt:full[1]=full[0];full[2]=0
            return full
        sp=obj.sparsity(active).tolil()
        if zero_tilt and 0 in active:sp[:,list(active).index(0)]=1
        fit=least_squares(lambda y:obj(expand(y)),x[active],bounds=(low[active],high[active]),
                          jac_sparsity=None if stage=='joint' else sp.tocsr(),x_scale='jac',loss='linear',max_nfev=1000,
                          ftol=1e-10,xtol=1e-10,gtol=1e-10,tr_options={'atol':1e-11,'btol':1e-11})
        x=expand(fit.x);Ks,cs,ds,ps=obj.unpack(x)
        stages.append(dict(stage=stage,success=bool(fit.success),message=fit.message,optimality=float(fit.optimality),nfev=fit.nfev,rms_px=float(np.sqrt(np.mean(obj(x).reshape(-1,2)**2)*2)),K=Ks,c=cs,kappa=ds))
        solutions.append(x.copy())
    refinement=None
    if polish_nfev is not None:
        from .solver_search import refine
        frozen=x.copy()
        def expanded(y):
            full=frozen.copy();full[active]=y
            if zero_tilt:full[1]=full[0];full[2]=0
            return full
        y,refinement=refine(lambda y:obj(expanded(y)),x[active],low[active],high[active],max_nfev=polish_nfev)
        x=expanded(y)
        if refinement['accepted']:
            Ks,cs,ds,_=obj.unpack(x)
            stages.append(dict(stage='polish',success=True,message=refinement['message'],optimality=refinement['optimality'],
                nfev=refinement['nfev'],rms_px=float(np.sqrt(np.mean(np.sum(obj(x).reshape(-1,2)**2,axis=1)))),K=Ks,c=cs,kappa=ds))
    K,c,kappa,poses=obj.unpack(x);diag={}
    if diagnostics:
        def diagnostic_fun(z):
            z=z.copy()
            if zero_tilt:z[1]=z[0];z[2]=0
            return obj(z)
        J=finite_jacobian(diagnostic_fun,x,active)
        ng=sum(active<8);Jg=J[:,:ng];Jp=J[:,ng:]
        U,s,_=np.linalg.svd(Jp,full_matrices=False);U=U[:,s>s[0]*1e-9]
        diag=matrix_diagnostics(Jg-U@(U.T@Jg),[obj.names[i] for i in active[:ng]])
    near=[obj.names[i] for i in active if min(x[i]-low[i],high[i]-x[i])<1e-5]
    return dict(K=K,c=c,kappa=kappa,q_scale_mm=scale,poses={f['id']:pose for f,pose in zip(frames,poses)},
                initialization=init,stages=stages,global_diagnostics=diag,at_bounds=near,
                residuals=obj(x).reshape(-1,2),zero_tilt=zero_tilt,
                center_mode='estimated_with_radial' if release_center else 'fixed_geometric_center',
                refinement=refinement,
                tz_observable=False,pose_depth_sign_resolved=False,independent_angle_validation=False)


def calibrate_single_search(frames,shape,settings,release_center=True,zero_tilt=False):
    from .solver_search import robust_cost,candidate_key,global_values,stability
    if settings['profile']=='baseline':
        return calibrate_single(frames,shape,release_center=release_center,zero_tilt=zero_tilt)
    K,_=intrinsic_init(frames);c=(np.array(shape,float)-1)/2
    rng=np.random.default_rng(settings['seed']);records=[];best=None;best_record=None
    for trial in range(settings['starts']):
        initial=dict(K=K.copy(),c=c.copy())
        if trial:
            initial['K'][0,0]*=np.exp(rng.normal(0,.01));initial['K'][1,1]*=np.exp(rng.normal(0,.01))
            if not zero_tilt:initial['K'][1,0]+=np.sqrt(abs(K[0,0]*K[1,1]))*rng.normal(0,.002)
            if release_center:initial['c']+=rng.normal(0,20,2)
        print('[solver] single-camera start %d/%d'%(trial+1,settings['starts']),flush=True)
        try:
            result=calibrate_single(frames,shape,release_center=release_center,zero_tilt=zero_tilt,
                initial_override=initial,polish_nfev=settings['refine_nfev'])
            record=dict(trial=trial,score=robust_cost(result['residuals']),geometry_valid=bool(np.isfinite(result['residuals']).all()),
                converged=bool(result['stages'][-1]['success']),at_bounds=result['at_bounds'],
                values=global_values(result['K'],result['c']),initial_K=initial['K'].tolist(),initial_c=initial['c'].tolist(),
                refinement=result['refinement'],kappa=result['kappa'].tolist(),stages=result['stages'])
            records.append(record)
            if best is None or candidate_key(record)<candidate_key(best_record):best,best_record=result,record
        except (ValueError,np.linalg.LinAlgError) as exc:
            records.append(dict(trial=trial,failed=str(exc),geometry_valid=False,converged=False))
    if best is None:raise ValueError('All single-camera starts failed: '+str(records))
    best['optimizer_search']=stability(records,best_record['trial'],settings['seed'])
    best['optimizer_search'].update(loss='linear',f_scale=1.,common_parameter_bounds=True,heldout_used_for_selection=False,
        perturbations=dict(log_abs_J_K_sigma=.01,L_over_reference_sigma=.002,center_sigma_px=20 if release_center else 0),
        refine_nfev=settings['refine_nfev'])
    best['optimizer_settings']=settings
    return best

def validate_single(result,frames):
    reports=[];errors=[]
    for f in frames:
        # Deterministic spatially interleaved fit/evaluation points.
        mask=np.array(f['ids'])%2==0
        pose=fit_pose(f,result['K'],result['c'],result['kappa'],result['q_scale_mm'],mask)
        err=projection(f['P'][~mask],pose,result['K'],result['c'],result['kappa'],result['q_scale_mm'])-f['p'][~mask]
        errors.extend(err);reports.append(dict(image_id=f['id'],count=len(err),rms_px=float(np.sqrt(np.mean(np.sum(err*err,axis=1))))))
    return dict(per_frame=reports,rms_px=float(np.sqrt(np.mean(np.sum(np.array(errors)**2,axis=1)))) if errors else None)
