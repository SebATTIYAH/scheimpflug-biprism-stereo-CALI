"""Conditional interpretation, not mechanical certification."""
import numpy as np

def effective_angles(K,pixel=(1.,1.)):
    K=np.asarray(K,float).copy();pixel=np.asarray(pixel,float)
    if K.shape!=(2,2) or not np.isfinite(K).all():raise ValueError('Expected finite 2x2 K')
    if pixel.shape!=(2,) or not np.isfinite(pixel).all() or np.any(pixel<=0):raise ValueError('Pixel pitches must be two positive finite numbers')
    K[1]*=pixel[1]/pixel[0]
    j,k,l=K[0,0],K[1,1],K[1,0]
    if abs(K[0,1])>1e-10 or j*k<=0:raise ValueError('Expected signed lower-triangular K')
    r,h=k/j,l/j;B=h*h+r*r+1
    x=2/(B+np.sqrt((h*h+(r-1)**2)*(h*h+(r+1)**2)))
    alpha=np.arccos(np.sqrt(np.clip(x,0,1)));m=k*np.cos(alpha)
    beta=np.arctan(-l/(m*np.tan(alpha))) if abs(alpha)>1e-9 else np.arccos(np.clip(m/j,-1,1))
    if abs(h)<1e-12:
        if r>=1:alpha=np.arccos(1/r);beta=0.;m=j
        else:alpha=0.;beta=np.arccos(r);m=k
    singular=np.linalg.svd(K,compute_uv=False)
    theta=np.arccos(np.clip(singular[-1]/singular[0],0,1))
    return dict(alpha_deg=float(np.degrees(alpha)),beta_deg=float(np.degrees(beta)),
                total_tilt_deg=float(np.degrees(theta)),m_px_per_mm=float(m),
                status='conditional_effective_angle_not_independently_measured',
                sign_convention='alpha >= 0; (-alpha,-beta) is the equivalent unresolved sign branch')
