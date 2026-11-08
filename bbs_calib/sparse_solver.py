"""Block-sparse central differences and pose-block rank diagnostics (SciPy 1.10)."""
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, block_diag, vstack
from scipy.spatial.transform import Rotation


class GroupedJacobian:
    """Each frame has six local parameters; global columns never share a group."""
    def __init__(self,fun,pattern,active,workers=1,low=None,high=None,label='solve',n_global=8):
        self.fun=fun;self.pattern=csc_matrix(pattern);self.active=np.asarray(active)
        self.low=np.full(len(active),-np.inf) if low is None else np.asarray(low)
        self.high=np.full(len(active),np.inf) if high is None else np.asarray(high)
        self.pool=ThreadPoolExecutor(max_workers=workers) if workers>1 else None
        groups={}
        for j,k in enumerate(active):groups.setdefault(('global',int(k)) if k<n_global else ('pose',int((k-n_global)%6)),[]).append(j)
        self.groups=list(groups.values());self.label=label;self.calls=0;self.started=time.perf_counter();self.last=self.started
        # Verify the advertised independence instead of silently returning a wrong Jacobian.
        for group in self.groups:
            occupied=[]
            for j in group:occupied.extend(self.pattern.indices[self.pattern.indptr[j]:self.pattern.indptr[j+1]])
            if len(set(occupied))!=len(occupied):raise ValueError('Overlapping rows in finite-difference color group')

    def close(self):
        if self.pool:self.pool.shutdown(wait=True)

    def __call__(self,x):
        x=np.asarray(x);h=np.cbrt(np.finfo(float).eps)*np.maximum(1.,np.abs(x))
        hp=h.copy();hm=-h.copy()
        for j in range(len(x)):
            if x[j]+h[j]>self.high[j] or x[j]-h[j]<self.low[j]:
                if self.high[j]-x[j]>=x[j]-self.low[j]:hp[j]=min(h[j],self.high[j]-x[j]);hm[j]=0.
                else:hp[j]=0.;hm[j]=-min(h[j],x[j]-self.low[j])
        if np.any(hp-hm<=0):raise ValueError('No finite-difference room within parameter bounds')
        def evaluate(group):
            xp=x.copy();xm=x.copy();xp[group]+=hp[group];xm[group]+=hm[group]
            delta=self.fun(xp)-self.fun(xm);parts=[]
            for j in group:
                rows=self.pattern.indices[self.pattern.indptr[j]:self.pattern.indptr[j+1]]
                parts.append((j,delta[rows]/(xp[j]-xm[j])))
            return parts
        batches=list(self.pool.map(evaluate,self.groups)) if self.pool else [evaluate(g) for g in self.groups]
        values=np.empty(self.pattern.nnz)
        for batch in batches:
            for j,v in batch:values[self.pattern.indptr[j]:self.pattern.indptr[j+1]]=v
        self.calls+=1;now=time.perf_counter()
        if now-self.last>=20:
            print('[solver] %s: Jacobian %d, elapsed %.1f s'%(self.label,self.calls,now-self.started),flush=True);self.last=now
        return csc_matrix((values,self.pattern.indices,self.pattern.indptr),shape=self.pattern.shape).tocsr()


def vector_raw(codec,data,x,penalty=True):
    from .geometry import optical_system,path_validity
    from .projection import project
    state,poses=codec.unpack(x);rot=Rotation.from_rotvec(poses[:,:3]).as_matrix()
    X=np.einsum('nij,nj->ni',rot[data['frame']],data['P'])+poses[data['frame'],3:]
    system=optical_system(state['prism'],state['omega']);pred=np.empty_like(data['p']);invalid=0
    for branch in ('L','R'):
        mask=data['s']==branch;part=project(X[mask],branch,state,system)
        if penalty:
            valid,violation,_=path_validity(X[mask],branch,system,state['prism'])
            if not valid.all():
                invalid+=1;part[~valid]=data['p'][mask][~valid]+1000.+1000*violation[~valid,None]
        pred[mask]=part
    return pred-data['p'],invalid


def block_rank(J,active,frame_indices,names,extra_global_rows=None,n_global=8):
    from .optimization import matrix_diagnostics
    global_cols=np.flatnonzero(active<n_global);reduced=[];ranks=[]
    for frame in np.unique(frame_indices):
        rows=(2*np.flatnonzero(frame_indices==frame)[:,None]+[0,1]).ravel()
        local=np.flatnonzero((active>=n_global+6*frame)&(active<n_global+6*(frame+1)))
        A=J[rows][:,global_cols].toarray();B=J[rows][:,local].toarray()
        if len(local):
            U,s,_=np.linalg.svd(B,full_matrices=False);rank=int(np.sum(s>s[0]*1e-6)) if len(s) else 0
            Q=U[:,:rank];A=A-Q@(Q.T@A)
        else:rank=0;s=np.array([])
        ranks.append(dict(frame_index=int(frame),rank=rank,columns=len(local),singular_values=s))
        reduced.append(A)
    if extra_global_rows is not None:reduced.append(extra_global_rows)
    if len(global_cols):
        g=matrix_diagnostics(np.vstack(reduced),[names[j] for j in global_cols])
        reference=np.linalg.norm(J[:,global_cols].data)
        if extra_global_rows is not None:reference=max(reference,np.linalg.norm(extra_global_rows))
        threshold=max(reference,1.)*1e-6
        g['rank']=int(np.sum(np.asarray(g['singular_values'])>threshold))
        g['full_column_rank']=g['rank']==len(global_cols)
        g['rank_absolute_threshold']=threshold
        g['rank_threshold_note']='1e-6 times max(1, pre-elimination global Frobenius norm, prior norm); avoids promoting cancellation roundoff to rank'
        if not g['full_column_rank']:g['local_parameter_correlation']=None
    else:g=dict(rank=0,columns=0,full_column_rank=True,singular_values=[],high_column_correlations=[])
    rank=sum(r['rank'] for r in ranks)+g['rank']
    result=dict(rank=rank,columns=len(active),full_column_rank=rank==len(active),
        method='Per-frame pose SVD (relative 1e-6) + global SVD (pre-elimination scale-aware cutoff); not full-Jacobian SVD',
        rank_relative_threshold=1e-6,pose_blocks=ranks,scaled_parameter_names=names,
        singular_values=None,local_parameter_correlation=None,parameter_standard_deviations=None,
        full_spectrum_computed=False,uncertainty_note='Block rank is not full-Jacobian spectrum or metrological precision; thresholds differ from legacy full SVD')
    return result,g


def diagnostics(objective,x,active):
    data=objective.data;pattern=objective.sparsity(active)[:2*len(data['p'])]
    frozen=x.copy()
    def fun(y):
        full=frozen.copy();full[active]=y;return objective.raw(full,False).ravel()
    ng=objective.codec.n_global
    jac=GroupedJacobian(fun,pattern,active,workers=objective.codec.c['optimization'].get('workers',1),label='diagnostics',n_global=ng)
    try:J=jac(x[active])
    finally:jac.close()
    names=[objective.codec.names[i] for i in active]
    raw,g=block_rank(J,active,data['frame'],names,n_global=ng)
    # Whiten each observation without computing a second numerical Jacobian.
    W=block_diag(list(data['W']),format='csr');weighted=W@J
    priors=np.zeros((len(objective.priors),sum(active<ng)))
    global_positions={int(k):j for j,k in enumerate(active[active<ng])}
    for row,(idx,mean,sigma) in enumerate(objective.priors):
        if idx in global_positions:priors[row,global_positions[idx]]=1/sigma
    augmented,_=block_rank(weighted,active,data['frame'],names,priors,n_global=ng)
    raw['jacobian_nnz']=int(J.nnz);raw['jacobian_storage_bytes']=int(J.data.nbytes+J.indices.nbytes+J.indptr.nbytes)
    return raw,augmented,g
