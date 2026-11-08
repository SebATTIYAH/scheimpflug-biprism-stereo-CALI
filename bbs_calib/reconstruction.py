from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .geometry import optical_system, path_validity, GeometryError
from .projection import project


def undistort(qd: np.ndarray, kappa: np.ndarray, scale: float, radius_limit_mm: float) -> np.ndarray:
    """Invert radial radius only on the declared monotone field of view."""
    k1, k2 = kappa
    top = (radius_limit_mm/scale)**2
    locations = [0., top]
    if k2 != 0 and 0 < -3*k1/(10*k2) < top:
        locations.append(-3*k1/(10*k2))
    if min(1+3*k1*s+5*k2*s*s for s in locations) <= 1e-8:
        raise GeometryError('Radial mapping is not monotone on declared field of view')
    def radial(r: float) -> float:
        s = (r/scale)**2
        return r*(1+k1*s+k2*s*s)
    result = []
    for v in np.asarray(qd, float).reshape(-1, 2):
        rd = np.linalg.norm(v)
        if rd > radial(radius_limit_mm)+1e-10:
            raise GeometryError('Pixel outside declared invertible distortion field')
        r = brentq(lambda z: radial(z)-rd, 0., radius_limit_mm, xtol=1e-13) if rd > 1e-15 else 0.
        if abs(radial(r)-rd) > 1e-9:
            raise GeometryError('Radial inversion did not converge')
        result.append(v*(r/rd) if rd else v.copy())
    return np.asarray(result)


def reconstruct_points(left_pixels: np.ndarray, right_pixels: np.ndarray, calibration: dict,
                       pixel_coordinates: str = 'full_frame', image_id: str = None) -> dict:
    """Inputs follow IMAGE left/right views; config maps these to physical L/R."""
    state = calibration.get('state', calibration)
    state = dict(state)
    for key in ('K','c','omega','kappa'):
        state[key] = np.asarray(state[key], float)
    if left_pixels is None or right_pixels is None:
        raise ValueError('Both views are required for 3D reconstruction')
    inputs = [np.asarray(v, float).reshape(-1, 2) for v in (left_pixels,right_pixels)]
    if pixel_coordinates=='crop_local':
        from .image_inputs import sensor_pixels
        inputs=[sensor_pixels(p,calibration,v,image_id) for p,v in zip(inputs,('left','right'))]
    elif pixel_coordinates!='full_frame':
        raise ValueError('pixel_coordinates must be full_frame or crop_local')
    if inputs[0].shape != inputs[1].shape or not all(np.isfinite(v).all() for v in inputs):
        raise ValueError('Matched finite pairs with identical shape are required')
    mapping = calibration.get('view_branches', dict(left='L', right='R'))
    system = optical_system(state['prism'], state['omega'])
    branches = [mapping['left'], mapping['right']]
    if set(branches) != {'L','R'}:
        raise ValueError('Invalid branch mapping')
    H = np.vstack([system[s]['M'] for s in branches])
    sv = np.linalg.svd(H, compute_uv=False)
    rank = int(np.sum(sv > sv[0]*1e-10))
    condition = float(sv[0]/sv[-1]) if sv[-1] > 0 else float('inf')
    if rank < 3 or condition > 1e7:
        return dict(X_mm=None, geometry_valid=False, rank=rank, singular_values=sv, condition=condition,
                    reason='Stereo depth is degenerate or severely ill-conditioned')
    rhs = []
    for p, s in zip(inputs, branches):
        qd = np.linalg.solve(state['K'], (p-state['c']).T).T
        q = undistort(qd, state['kappa'], state['q_scale'], calibration.get('radius_limit_mm', 2*state['q_scale']))
        rhs.append(q-system[s]['b'])
    X = np.linalg.lstsq(H, np.hstack(rhs).T, rcond=None)[0].T
    valid = np.ones(len(X), bool)
    reasons = []
    errors = []
    for s, p in zip(branches, inputs):
        good, _, why = path_validity(X, s, system, state['prism'])
        valid &= good
        reasons.append(why)
        errors.append(project(X, s, state, system)-p)
    return dict(X_mm=X, valid_mask=valid, geometry_valid=bool(valid.all()),
                rank=rank, singular_values=sv, condition=condition,
                reprojection_errors_px=np.stack(errors, axis=1), reasons=reasons,
                aperture_verified=state['prism'].get('aperture_mm') is not None)
