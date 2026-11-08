"""Translated bi-telecentric biprism bundle adjustment (separate model version).

X and board poses are in the prism frame. Its transverse translation relative
to the camera reference is delta*n, where n is normal to the projected ridge.
Along-ridge translation and camera-axis translation are coordinate gauges.
The common translation is added BEFORE radial distortion, in BOTH projection
and inverse reconstruction. Prism interfaces stay in their own local frame.
"""
from copy import deepcopy
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from .geometry import optical_system,path_validity
from .projection import project as base_project,object_points,radial_coefficients
from .reconstruction import undistort,undistort_state
from .brown_distortion import has_brown,parameters as brown_parameters,certificate
from .initialization import initialize_pose
from .optimization import Parameters
from .sparse_solver import GroupedJacobian,block_rank
from .validation import error_metrics

MODEL='bbs-transverse-bundle-v2'
MODEL_RADIAL3='bbs-transverse-bundle-radial3-v3'
MODEL_BROWN='bbs-transverse-bundle-brown-v4'
SUPPORTED_MODELS=(MODEL,MODEL_RADIAL3,MODEL_BROWN)


def model_for_state(state):
    if has_brown(state):
        brown_parameters(state)
        return MODEL_BROWN
    return MODEL if len(radial_coefficients(state['kappa'])) == 2 else MODEL_RADIAL3


def validate_model(cal):
    if cal.get('model_version') not in SUPPORTED_MODELS:
        raise ValueError('Expected translated v2, radial3 v3 or Brown v4 calibration')
    if cal['model_version'] != model_for_state(cal['state']):
        raise ValueError('Model version and radial coefficient count disagree')


def translated_system(state):
    system=optical_system(state['prism'],state['omega'])
    tangent=system['Rp'][:2,1]
    normal=np.array([tangent[1],-tangent[0]])
    normal/=np.linalg.norm(normal)
    if normal[0]<0:normal=-normal
    shift=normal*float(state.get('transverse_offset_mm',0.))
    system['transverse_normal_camera_xy']=normal
    system['camera_translation_xy_mm']=shift
    for branch in ('L','R'):
        system[branch]['b']=system[branch]['b']+shift
        system[branch]['G']=system[branch]['G'].copy()
        system[branch]['G'][:,3]+=shift
    return system


def project(X,branch,state,system=None):
    return base_project(X,branch,state,system or translated_system(state))


def pose_initial(P,pixels,branches,state):
    system=translated_system(state)
    q=undistort_state(np.linalg.solve(state['K'],(pixels-state['c']).T).T,state)
    q-=system['camera_translation_xy_mm']
    linear=deepcopy(state);linear['kappa']=np.zeros_like(state['kappa'])
    linear.pop('tangential',None);linear.pop('distortion_center_px',None)
    pose,_=initialize_pose(P,q@state['K'].T+state['c'],branches,linear)
    branches=np.asarray(branches)
    def fun(x):
        X=object_points(P,x);pred=np.empty_like(pixels)
        for b in ('L','R'):
            m=branches==b;pred[m]=project(X[m],b,state,system)
        return (pred-pixels).ravel()
    result=least_squares(fun,pose,x_scale=[.1,.1,.1,1,1,1],max_nfev=150)
    if not result.success:raise ValueError('Per-frame pose initialization did not converge')
    return result.x


class BundleParameters(Parameters):
    def __init__(self,state,frames,c):
        super().__init__(state,frames,c)
        if self.hardware:raise ValueError('v2 holds prism material/angles/thickness fixed')
        self.n_radial=len(radial_coefficients(state['kappa']))
        self.offset_index=6+self.n_radial
        self.n_global=self.offset_index+1
        self.brown=has_brown(state)
        extra=[]
        if self.brown:
            brown_parameters(state)
            self.tangential_slice=slice(self.n_global,self.n_global+2)
            self.center_slice=slice(self.n_global+2,self.n_global+4)
            extra=['p1','p2','distortion_center_x_delta_over_100px','distortion_center_y_delta_over_100px']
            self.n_global+=4
        self.names=(self.names[:6]+['kappa%d'%(i+1) for i in range(self.n_radial)]
                    +['transverse_offset_mm']+extra+self.names[8:])

    def pack(self,state,poses):
        k=state['K']
        extra=np.r_[state['tangential'],(np.asarray(state['distortion_center_px'])-self.base['c'])/100.] if self.brown else []
        return np.r_[np.log(abs(k[0,0])),np.log(abs(k[1,1])),k[1,0]/self.scale,
                     state['omega']/.1,state['kappa'],state.get('transverse_offset_mm',0.),
                     extra,
                     (poses/[.1,.1,.1,1,1,1]).ravel()]

    def unpack(self,x):
        state=deepcopy(self.base)
        state['K']=np.array([[self.sign[0]*np.exp(x[0]),0.],[x[2]*self.scale,self.sign[1]*np.exp(x[1])]])
        state['omega']=x[3:6]*.1;state['kappa']=x[6:self.offset_index].copy()
        state['transverse_offset_mm']=float(x[self.offset_index])
        if self.brown:
            state['tangential']=x[self.tangential_slice].copy()
            state['distortion_center_px']=np.asarray(self.base['c'])+100*x[self.center_slice]
        return state,x[self.n_global:].reshape(-1,6)*[.1,.1,.1,1,1,1]

    def bounds(self,x):
        lo,hi=super().bounds(x)
        bound=self.c['optimization']['transverse_offset_bound_mm']
        lo[6:self.offset_index],hi[6:self.offset_index]=self.c['optimization'].get('radial_bounds',[-.15,.15])
        lo[self.offset_index],hi[self.offset_index]=-bound,bound
        if self.brown:
            opt=self.c['optimization']
            pbound=float(opt.get('tangential_bound',.05))
            cbound=float(opt.get('distortion_center_bound_px',300.))/100.
            if not np.isfinite([pbound,cbound]).all() or min(pbound,cbound)<=0:
                raise ValueError('Positive finite tangential/center search bounds required')
            lo[self.tangential_slice],hi[self.tangential_slice]=-pbound,pbound
            lo[self.center_slice],hi[self.center_slice]=-cbound,cbound
        return lo,hi

    def fit_indices(self):
        fixed=set()
        if self.c['optimization'].get('fixed_offset_mm') is not None:fixed.add(self.offset_index)
        if self.brown and not self.c['optimization'].get('release_distortion_center',True):
            fixed.update(range(self.center_slice.start,self.center_slice.stop))
        return np.array([i for i in range(len(self.names)) if i not in fixed],int)


class BundleObjective:
    def __init__(self,codec,data):
        self.codec,self.data=codec,data
        self.weight=float(codec.c['optimization']['path_weight_px_per_mm'])

    def evaluate(self,x):
        state,poses=self.codec.unpack(x)
        system=translated_system(state);d=self.data
        rotations=Rotation.from_rotvec(poses[:,:3]).as_matrix()
        X=np.einsum('nij,nj->ni',rotations[d['frame']],d['P'])+poses[d['frame'],3:]
        pred=np.empty_like(d['p']);violation=np.empty(len(pred))
        for b in ('L','R'):
            mask=d['s']==b;pred[mask]=project(X[mask],b,state,system)
            violation[mask]=path_validity(X[mask],b,system,state['prism'])[1]
        return pred-d['p'],violation

    def __call__(self,x):
        raw,v=self.evaluate(x)
        return np.r_[np.einsum('nij,nj->ni',self.data['W'],raw).ravel(),self.weight*v]

    def sparsity(self,active):
        n=len(self.data['p']);J=lil_matrix((3*n,len(self.codec.names)),dtype=int)
        ng=self.codec.n_global
        J[:,:ng]=1
        for i,f in enumerate(self.data['frame']):
            J[2*i:2*i+2,ng+6*f:ng+6*f+6]=1
            J[2*n+i,ng+6*f:ng+6*f+6]=1
        return J[:,active].tocsr()

    def loss(self,z):
        # Robust pixel rows, ordinary quadratic geometry rows. f_scale is px.
        rho=np.vstack([z.copy(),np.ones_like(z),np.zeros_like(z)])
        m=2*len(self.data['p']);q=1+z[:m]
        rho[0,:m]=2*(np.sqrt(q)-1);rho[1,:m]=1/np.sqrt(q)
        rho[2,:m]=-.5/q**1.5
        return rho


def fit_once(data,state,c,label):
    state=deepcopy(state)
    fixed_offset=c['optimization'].get('fixed_offset_mm')
    if fixed_offset is not None:state['transverse_offset_mm']=float(fixed_offset)
    poses=[]
    for i,f in enumerate(data['frames']):
        mask=data['frame']==i
        poses.append(pose_initial(data['P'][mask],data['p'][mask],data['s'][mask],state))
    codec=BundleParameters(state,data['frames'],c)
    x=codec.pack(state,np.array(poses));lo,hi=codec.bounds(x)
    if np.any(x<=lo)|np.any(x>=hi):raise ValueError('Initial parameters outside bounds')
    obj=BundleObjective(codec,data);active=codec.fit_indices()
    def expand(y):
        full=x.copy();full[active]=y;return full
    fun=lambda y:obj(expand(y))
    jac=GroupedJacobian(fun,obj.sparsity(active),active,c['optimization'].get('workers',1),
                        lo[active],hi[active],label=label,n_global=codec.n_global)
    opt=c['optimization'];f=float(opt.get('f_scale_px',.5))
    try:
        r=least_squares(fun,x[active],jac=jac,method='trf',tr_solver='lsmr',bounds=(lo[active],hi[active]),
            x_scale='jac',loss=obj.loss,f_scale=f,max_nfev=opt.get('max_nfev',400),
            ftol=1e-9,xtol=1e-10,gtol=1e-8,tr_options={'atol':1e-11,'btol':1e-11})
    finally:jac.close()
    solution=expand(r.x)
    fitted,poses=codec.unpack(solution);raw,v=obj.evaluate(solution)
    invertible=True;cert=None
    if has_brown(fitted):
        cert=certificate(fitted);invertible=cert['certified']
    else:
        try:undistort(np.zeros((1,2)),fitted['kappa'],fitted['q_scale'],fitted['radius_limit_mm'])
        except ValueError:invertible=False
    diagnostic=dict(success=bool(r.success),status=int(r.status),message=r.message,nfev=r.nfev,
        optimality=float(r.optimality),objective=float(r.cost),training=error_metrics(raw),
        invalid_paths=int(np.sum(v>1e-8)),max_path_violation_mm=float(v.max()),
        geometry_valid=bool(np.all(v<=1e-8)),distortion_invertible=invertible,distortion_certificate=cert,
        at_bounds=[codec.names[i] for i in active if min(solution[i]-lo[i],hi[i]-solution[i])<1e-4],
        transverse_offset_mm=fitted['transverse_offset_mm'],state=fitted)
    return fitted,poses,diagnostic,(codec,obj,solution)


def data_rank(codec,obj,x):
    active=codec.fit_indices();pattern=obj.sparsity(active)[:2*len(obj.data['p'])]
    def raw(y):
        full=x.copy();full[active]=y
        return obj.evaluate(full)[0].ravel()
    jac=GroupedJacobian(raw,pattern,active,1,n_global=codec.n_global,label='data rank')
    try:J=jac(x[active])
    finally:jac.close()
    full,global_diag=block_rank(J,active,obj.data['frame'],[codec.names[i] for i in active],n_global=codec.n_global)
    return dict(full=full,global_after_eliminating_poses=global_diag,
        fixed_parameters=[n for i,n in enumerate(codec.names) if i not in set(active)],
        note='Unweighted pixel-only Jacobian, excludes geometry constraints; rank is local and conditional, not absolute parameter accuracy.')


def distortion_identifiability(rank,state):
    if not has_brown(state):return None
    gd=rank['global_after_eliminating_poses']
    terms={'p1','p2','distortion_center_x_delta_over_100px','distortion_center_y_delta_over_100px'}
    pairs=[r for r in gd['high_column_correlations'] if r['a'] in terms or r['b'] in terms]
    return dict(weakly_identified=bool(pairs) or not gd['full_column_rank'],
                high_sensitivity_correlations=pairs,
                warning='Bounds are search limits, not priors or confidence intervals. Center and tangential terms may trade off with prism pose/offset and radial terms.')


def reconstruct(left,right,state,mapping):
    if set(mapping.values())!={'L','R'}:raise ValueError('Invalid branch mapping')
    left,right=np.asarray(left,float),np.asarray(right,float)
    if left.shape!=right.shape or left.ndim!=2 or left.shape[1]!=2 or not np.isfinite([left,right]).all():
        raise ValueError('Finite matched Nx2 full-sensor pixels required')
    system=translated_system(state);branches=[mapping[v] for v in ('left','right')]
    H=np.vstack([system[b]['M'] for b in branches]);sv=np.linalg.svd(H,compute_uv=False)
    if sv[-1]<sv[0]*1e-10:raise ValueError('Degenerate stereo geometry')
    rhs=[]
    for pixels,b in zip((left,right),branches):
        q=undistort_state(np.linalg.solve(state['K'],(pixels-state['c']).T).T,state)
        rhs.append(q-system[b]['b'])
    X=np.linalg.lstsq(H,np.hstack(rhs).T,rcond=None)[0].T
    valid=np.ones(len(X),bool);errors=[]
    for pixels,b in zip((left,right),branches):
        valid &= path_validity(X,b,system,state['prism'])[0]
        errors.append(project(X,b,state,system)-pixels)
    return dict(X_mm=X,valid_mask=valid,geometry_valid=bool(valid.all()),
                condition=float(sv[0]/sv[-1]),reprojection_errors_px=np.stack(errors,axis=1))


def validate_frames(obs,frames,state,c):
    rng=np.random.default_rng(c.get('seed',1234)+29);reports=[];errors=[]
    mapping={v:c['views'][v]['branch'] for v in ('left','right')}
    for frame in frames:
        rows=[r for r in obs['used'] if r['row']['image_id']==frame]
        ids=sorted({r['row']['point_id'] for r in rows})
        fitting=set(rng.choice(ids,max(4,len(ids)//2),replace=False).tolist())
        fit=[r for r in rows if r['row']['point_id'] in fitting];test=[r for r in rows if r['row']['point_id'] not in fitting]
        try:
            pose=pose_initial(np.array([r['P'] for r in fit]),np.array([r['p'] for r in fit]),[r['branch'] for r in fit],state)
            raw=np.array([project(object_points(r['P'][None],pose),r['branch'],state)[0]-r['p'] for r in test])
            errors.extend(raw)
            views={v:{r['row']['point_id']:r for r in rows if r['row']['view']==v} for v in ('left','right')}
            paired=sorted(set(views['left'])&set(views['right']))
            rec=reconstruct(*[np.array([views[v][i]['p'] for i in paired]) for v in ('left','right')],state,mapping)
            report=dict(image_id=frame,status='evaluated',metrics=error_metrics(raw),
                        geometry_valid=rec['geometry_valid'],invalid_points=int(np.sum(~rec['valid_mask'])),
                        pose=pose,pose_fit_point_ids=sorted(fitting),
                        test_point_ids=sorted(set(ids)-fitting))
            sel=np.array([i not in fitting for i in paired])
            if rec['valid_mask'][sel].all():
                X=rec['X_mm'][sel];P=np.array([views['left'][i]['P'] for i in paired])[sel]
                dd=np.linalg.norm(X[:,None]-X[None],axis=2)-np.linalg.norm(P[:,None]-P[None],axis=2)
                report['pair_distance_rms_mm']=float(np.sqrt(np.mean(dd[np.triu_indices(len(X),1)]**2)))
                _,_,vt=np.linalg.svd(X-X.mean(axis=0))
                report['plane_rms_mm']=float(np.sqrt(np.mean(((X-X.mean(axis=0))@vt[-1])**2)))
            reports.append(report)
        except ValueError as exc:reports.append(dict(image_id=frame,status='failed',reason=str(exc),geometry_valid=False))
    return dict(metrics=error_metrics(errors),frames=reports,all_geometry_valid=all(r['geometry_valid'] for r in reports),
                warning='Held-out pose fit uses disjoint IDs; board self-consistency is not independent displacement/strain accuracy.')


def feasible_offset_interval(data,state):
    """Initial feasible interval from TRAINING measured rays only, not an estimate."""
    system=translated_system(state)
    q=undistort_state(np.linalg.solve(state['K'],(data['p']-state['c']).T).T,state)
    A=system['Rp'][:2,:2];back=np.linalg.solve(A,q.T).T
    normal=system['transverse_normal_camera_xy'];coefficient=np.linalg.solve(A,normal)[0]
    ridge=-state['prism']['thickness_mm']*system['d2'][0]/system['d2'][2]
    if coefficient<=0:raise ValueError('Unsupported offset-axis convention')
    left=data['s']=='L';right=data['s']=='R'
    return [(float(back[left,0].max())-ridge)/coefficient,
            (float(back[right,0].min())-ridge)/coefficient]
