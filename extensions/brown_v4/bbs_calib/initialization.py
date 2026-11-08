from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from .geometry import optical_system, nearest_rotation, GeometryError
from .projection import project, object_points
from .reconstruction import undistort


def initialize_pose(P: np.ndarray, pixels: np.ndarray, branches: list, state: dict) -> tuple:
    """Both refracting branches, explicit principal subtraction; planar/nonplanar paths."""
    system = optical_system(state['prism'], state['omega'])
    q = np.linalg.solve(state['K'], (pixels-state['c']).T).T
    q = undistort(q, state['kappa'], state['q_scale'], state.get('radius_limit_mm', 2*state['q_scale']))
    planar = np.max(np.abs(P[:, 2])) < 1e-10
    A, b = [], []
    for xyz, z, s in zip(P, q, branches):
        M = system[s]['M']
        A.append(np.hstack([xyz[j]*M for j in range(2 if planar else 3)]+[M]))
        b.append(z-system[s]['b'])
    A, b = np.vstack(A), np.concatenate(b)
    w, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    if rank < A.shape[1] or sv[0]/sv[-1] > 1e10:
        raise GeometryError('Pose initialization design is rank deficient; need matched views and noncollinear points')
    if planar:
        r1, r2 = w[:3], w[3:6]
        R = nearest_rotation(np.column_stack([r1, r2, np.cross(r1, r2)]))
    else:
        R = nearest_rotation(w[:9].reshape(3, 3).T)
    pose = np.r_[Rotation.from_matrix(R).as_rotvec(), w[-3:]]
    def residual(x: np.ndarray) -> np.ndarray:
        X = object_points(P, x)
        out = np.empty_like(pixels)
        for s in ('L','R'):
            mask = np.asarray(branches) == s
            out[mask] = project(X[mask], s, state, system)
        return (out-pixels).ravel()
    result = least_squares(residual, pose, method='trf', x_scale=np.r_[np.ones(3)*.1, np.ones(3)], max_nfev=150)
    return result.x, dict(design_rank=int(rank), design_condition=float(sv[0]/sv[-1]),
                          pose_fit_converged=bool(result.success))
