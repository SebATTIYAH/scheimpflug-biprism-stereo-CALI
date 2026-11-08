"""Centered Brown radial+tangential map on the telecentric q plane.

The fixed affine pixel origin c is NOT the fitted distortion center.
Center is stored in full-sensor pixels, d = inv(K) @ (center_px-c).
With v=(q-d)/scale, F(v)=v*R(r²)+[2*p1*x*y+p2*(r²+2*x²),
                                    p1*(r²+2*y²)+2*p2*x*y].
The valid domain is the disk |q-d| <= radius_limit_mm.
"""
import numpy as np
from .projection import radial_coefficients
from .geometry import GeometryError


def has_brown(state):
    keys=('tangential','distortion_center_px')
    present=[k in state for k in keys]
    if any(present) and not all(present):
        raise ValueError('Both tangential and distortion_center_px must be supplied')
    return all(present)


def parameters(state):
    k=radial_coefficients(state['kappa'])
    if not has_brown(state):raise ValueError('Missing centered Brown parameters')
    p=np.asarray(state['tangential'],float)
    center=np.asarray(state['distortion_center_px'],float)
    c=np.asarray(state['c'],float);K=np.asarray(state['K'],float)
    if p.shape!=(2,) or center.shape!=(2,) or c.shape!=(2,) or K.shape!=(2,2):
        raise ValueError('Invalid Brown parameter shape')
    if not all(np.isfinite(a).all() for a in (p,center,c,K)):
        raise ValueError('Nonfinite Brown parameters')
    scale=float(state['q_scale']);radius=float(state['radius_limit_mm'])
    if not np.isfinite([scale,radius]).all() or min(scale,radius)<=0:
        raise ValueError('Positive finite Brown scale/radius required')
    try:d=np.linalg.solve(K,center-c)
    except np.linalg.LinAlgError as exc:raise ValueError('Singular pixel matrix') from exc
    return k,p,d,scale,radius


def normalized_map(v,k,p,jacobian=False):
    v=np.asarray(v,float);x,y=v[...,0],v[...,1];s=x*x+y*y
    k=np.pad(k,(0,3-len(k)));k1,k2,k3=k;p1,p2=p
    radial=1+k1*s+k2*s*s+k3*s*s*s
    out=v*radial[...,None]+np.stack([2*p1*x*y+p2*(s+2*x*x),
                                     p1*(s+2*y*y)+2*p2*x*y],axis=-1)
    if not jacobian:return out
    grad=k1+2*k2*s+3*k3*s*s
    J=np.empty(v.shape[:-1]+(2,2))
    J[...,0,0]=radial+2*x*x*grad+2*p1*y+6*p2*x
    J[...,1,1]=radial+2*y*y*grad+6*p1*y+2*p2*x
    J[...,0,1]=J[...,1,0]=2*x*y*grad+2*p1*x+2*p2*y
    return out,J


def forward(q,state):
    k,p,d,scale,_=parameters(state)
    return d+scale*normalized_map((np.asarray(q,float)-d)/scale,k,p)


def polynomial_minimum(coeff,top):
    """Exact candidate locations of a cubic polynomial minimum on [0, top]."""
    poly=np.polynomial.Polynomial(coeff)
    points=[0.,top]
    for z in poly.deriv().roots():
        if abs(z.imag)<1e-12 and 0<z.real<top:points.append(float(z.real))
    return min(float(poly(t)) for t in points)


def certificate(state):
    """Sufficient global injectivity test, not a sampled determinant test.

    Radial J eigenvalues are R(s) and R(s)+2*s*R'(s).
    Tangential symmetric J has eigenvalues 4*(p2*x+p1*y) +/- 2*|p|*r;
    hence lambda_min(J) >= min(radial eigenvalues)-6*|p|*radius/scale.
    A positive bound implies strong monotonicity on the convex domain disk,
    hence injectivity there. Failure is inconclusive, not proof of a fold.
    """
    k,p,d,scale,radius=parameters(state)
    k=np.pad(k,(0,3-len(k)));R=radius/scale
    radial_min=min(polynomial_minimum(np.r_[1.,k],R*R),
                   polynomial_minimum(np.r_[1.,k*np.array([3.,5.,7.])],R*R))
    bound=radial_min-6*np.linalg.norm(p)*R
    return dict(certified=bool(np.isfinite(bound) and bound>1e-8),
                jacobian_eigenvalue_lower_bound=float(bound),radius_limit_mm=radius,
                domain_center_q_mm=d,domain_center_px=state['distortion_center_px'],
                method='Sufficient strong-monotonicity bound over full centered disk',
                failure_meaning='Uncertified; does not necessarily mean non-injective')


def inverse(qd,state):
    k,p,d,scale,radius=parameters(state)
    cert=certificate(state)
    if not cert['certified']:raise GeometryError('Brown distortion not certified invertible on declared disk')
    qd=np.asarray(qd,float)
    if qd.ndim!=2 or qd.shape[1]!=2 or not np.isfinite(qd).all():
        raise GeometryError('Finite Nx2 distorted coordinates required')
    if not len(qd):return qd.copy()
    if np.all(p==0):
        from .reconstruction import undistort
        return d+undistort(qd-d,k,scale,radius)
    target=(qd-d)/scale;R=radius/scale
    v=target.copy();norm=np.linalg.norm(v,axis=1)
    v*=np.minimum(1.,R/np.maximum(norm,1e-30))[:,None]
    tol=1e-10/scale
    for _ in range(60):
        predicted,J=normalized_map(v,k,p,True)
        residual=predicted-target;error=np.linalg.norm(residual,axis=1)
        active=error>tol
        if not active.any():break
        indices=np.flatnonzero(active)
        step=np.linalg.solve(J[active],residual[active])
        accepted=np.zeros(len(indices),bool);alpha=np.ones(len(indices))
        # Per-point damping, all iterates stay in the certified domain.
        for _ in range(35):
            todo=np.flatnonzero(~accepted)
            if not len(todo):break
            trial=v[indices[todo]]-alpha[todo,None]*step[todo]
            good=(np.linalg.norm(trial,axis=1)<=R+1e-12)
            trial_error=np.linalg.norm(normalized_map(trial,k,p)-target[indices[todo]],axis=1)
            good &= (trial_error<error[indices[todo]]) | (trial_error<=tol)
            chosen=todo[good];v[indices[chosen]]=trial[good];accepted[chosen]=True
            alpha[todo[~good]]*=.5
        if not accepted.all():
            raise GeometryError('Brown inverse failed or pixel outside declared mapped disk')
    check=np.linalg.norm(normalized_map(v,k,p)-target,axis=1)*scale
    if np.any(check>1e-9) or np.any(np.linalg.norm(v,axis=1)>R+1e-10):
        raise GeometryError('Brown inverse did not converge inside declared disk')
    return d+scale*v
