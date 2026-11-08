from __future__ import annotations
import warnings
import numpy as np
from scipy.spatial.transform import Rotation


class GeometryError(ValueError):
    pass


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-14:
        raise GeometryError('Invalid direction vector')
    return v/n


def snell(d: np.ndarray, N: np.ndarray, n_in: float, n_out: float) -> np.ndarray:
    """N points into the transmitted medium; propagation is along d."""
    d, N = unit(d), unit(N)
    if min(n_in, n_out) <= 0:
        raise GeometryError('Invalid refractive index')
    cosine = float(d @ N)
    if cosine < 1e-9:
        raise GeometryError('Normal orientation mismatch or grazing incidence')
    tangent = (n_in/n_out)*(d-cosine*N)
    root = 1-float(tangent @ tangent)
    if root < -1e-12:
        raise GeometryError('Total internal reflection')
    if root < 0:
        warnings.warn('Snell roundoff within 1e-12 clipped', RuntimeWarning)
        root = 0.
    if root < 1e-16:
        raise GeometryError('Grazing transmitted ray')
    return unit(tangent+np.sqrt(root)*N)


def plane_projection(d: np.ndarray, a: np.ndarray, h: float) -> np.ndarray:
    d, a = np.asarray(d, float), np.asarray(a, float)
    den = float(a @ d)
    if abs(den) < 1e-10:
        raise GeometryError('Ray parallel to interface')
    P = np.eye(4)
    P[:3, :3] -= np.outer(d, a)/den
    P[:3, 3] = h*d/den
    return P


def scheimpflug(alpha: float, beta: float) -> np.ndarray:
    if abs(alpha) >= np.pi/2 or abs(beta) >= np.pi/2:
        raise GeometryError('Sensor tilt must be below 90 degrees')
    return np.array([[1/np.cos(beta), 0.], [-np.tan(alpha)*np.tan(beta), 1/np.cos(alpha)]])


def optical_system(prism: dict, omega: np.ndarray) -> dict:
    """Recompute directions AND projections whenever Rp changes."""
    Rp = Rotation.from_rotvec(omega).as_matrix()
    d0 = Rp.T @ np.array([0., 0., 1.])
    d2 = snell(d0, np.array([0., 0., 1.]), 1., prism['refractive_index'])
    B2 = plane_projection(d2, np.array([0., 0., 1.]), 0.)
    Rp4 = np.eye(4)
    Rp4[:3, :3] = Rp
    result = dict(Rp=Rp, d0=d0, d2=d2, B2=B2)
    for s, sign in [('L', -1), ('R', 1)]:
        tau = sign*np.tan(np.radians(prism['wedge_'+s+'_deg']))
        a = np.array([tau, 0., 1.])
        d1 = snell(d2, a, prism['refractive_index'], 1.)
        B1 = plane_projection(d1, a, prism['thickness_mm'])
        B = B2 @ B1
        G = (Rp4 @ B)[:2]
        result[s] = dict(a=a, d1=d1, B1=B1, B=B, G=G, M=G[:, :3], b=G[:, 3])
    return result


def path_validity(X: np.ndarray, branch: str, system: dict, prism: dict) -> tuple:
    X = np.asarray(X, float).reshape(-1, 3)
    part = system[branch]
    front = np.column_stack([X, np.ones(len(X))]) @ part['B1'].T
    back = front @ system['B2'].T
    sign = -1 if branch == 'L' else 1
    # Distances in mm. Forward/backward line parameter alone is not a validity criterion.
    v = np.maximum(-sign*front[:, 0], 0.)
    v += np.maximum(-front[:, 2], 0.)
    v += np.maximum(prism['thickness_mm']-X @ part['a'], 0.)
    v += np.maximum(-np.sum((X-front[:, :3])*part['d1'], axis=1), 0.)
    v += np.maximum(-np.sum((front[:, :3]-back[:, :3])*system['d2'], axis=1), 0.)
    bounds = prism.get('aperture_mm')
    if bounds:
        for axis, key in enumerate(('x', 'y')):
            if key in bounds and bounds[key] is not None:
                lo, hi = bounds[key]
                for points in (front, back):
                    v += np.maximum(lo-points[:, axis], 0.)+np.maximum(points[:, axis]-hi, 0.)
    reasons = ['' if z <= 1e-8 else 'wrong_half/interface_order/aperture' for z in v]
    return v <= 1e-8, v, reasons


def nearest_rotation(A: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(A)
    return U @ np.diag([1., 1., np.linalg.det(U @ Vt)]) @ Vt
