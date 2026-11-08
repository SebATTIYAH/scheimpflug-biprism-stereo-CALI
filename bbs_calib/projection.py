from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import optical_system, path_validity


def distort(q: np.ndarray, kappa: np.ndarray, q_scale: float) -> np.ndarray:
    q = np.asarray(q, float)
    r2 = np.sum(q*q, axis=-1, keepdims=True)/q_scale**2
    return q*(1+kappa[0]*r2+kappa[1]*r2*r2)


def object_points(P: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return np.asarray(P) @ Rotation.from_rotvec(pose[:3]).as_matrix().T+pose[3:]


def project(X: np.ndarray, branch: str, state: dict, system: dict = None) -> np.ndarray:
    system = system or optical_system(state['prism'], state['omega'])
    g = system[branch]
    q = np.asarray(X) @ g['M'].T+g['b']
    return distort(q, state['kappa'], state['q_scale']) @ state['K'].T+state['c']


def projected_circle(P: np.ndarray, diameter: float, pose: np.ndarray,
                     branch: str, state: dict, samples: int = 128) -> np.ndarray:
    t = np.arange(samples)*2*np.pi/samples
    ring = P+diameter/2*np.column_stack([np.cos(t), np.sin(t), np.zeros(samples)])
    return project(object_points(ring, pose), branch, state)


def contour_review(boundary: np.ndarray, P: np.ndarray, diameter: float,
                   pose: np.ndarray, branch: str, state: dict) -> dict:
    """Optional model-dependent check, NOT independent center accuracy evidence."""
    from scipy.spatial import cKDTree
    ring = projected_circle(P, diameter, pose, branch, state, 720)
    distance, _ = cKDTree(ring).query(np.asarray(boundary))
    return dict(model_dependent=True, boundary_to_projected_ring_rms_px=float(np.sqrt(np.mean(distance**2))),
                p95_px=float(np.percentile(distance, 95)))
