"""Independent *candidate* shared-affine stereo camera.

The 3-D coordinates are in an effective metric camera frame, not the original
prism frame. A planar target does not independently anchor the depth sign or
surface normal; global rigid-frame and reflection conventions remain a gauge.
This module neither performs a Snell-law decomposition nor certifies
prism ray paths, mechanical Scheimpflug angles, or externally measured accuracy.
Do not pass this model to the legacy physical-camera reconstruction interface.

The affine matrix maps millimetres to UNDISTORTED full-sensor pixels. The fixed
legacy radial map is then applied independently to each image branch using the
same K, c and two radial coefficients. All public pixel arrays are row-major.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


MODEL_VERSION = "shared_affine_stereo_v1_candidate"
ROW_ORDER = ["left.u", "left.v", "right.u", "right.v"]
COORDINATE_WARNING = (
    "Effective metric camera coordinates, not the original physical prism frame. "
    "No prism-path validity, mechanical Scheimpflug angle, or external accuracy "
    "is established by this candidate model. Planar-target mirror ambiguity "
    "remains; axis signs do not establish physical depth. A valid solution is "
    "not a guarantee of reliable depth. Do not use the legacy interface."
)


def _array(value: Any, shape: tuple, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a numeric array" % name) from exc
    if result.shape != shape:
        raise ValueError("%s must have shape %s, got %s" % (name, shape, result.shape))
    if not np.all(np.isfinite(result)):
        raise ValueError("%s must contain only finite numbers" % name)
    return result


def _positive_scalar(value: Any, name: str) -> float:
    a = np.asarray(value)
    if a.ndim != 0 or isinstance(value, (bool, np.bool_)):
        raise ValueError("%s must be a positive finite scalar" % name)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a positive finite scalar" % name) from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError("%s must be a positive finite scalar" % name)
    return result


def _matrix_rank_and_condition(matrix: np.ndarray) -> tuple:
    singular = np.linalg.svd(matrix, compute_uv=False)
    tol = singular[0] * max(matrix.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > tol))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    return rank, condition


def _crop_origins(value: Any) -> Dict[str, np.ndarray]:
    if not isinstance(value, dict) or set(value) != {"left", "right"}:
        raise ValueError("crop_origins_px must specify exactly left and right")
    return {view: _array(value[view], (2,), "crop_origins_px.%s" % view)
            for view in ("left", "right")}


def validate_model(model: dict) -> None:
    """Reject incompatible models, malformed matrices and non-invertible radii.

    Radial monotonicity is checked analytically over the ENTIRE declared domain,
    including a possible interior derivative minimum, not on a sampling grid.
    """
    if not isinstance(model, dict):
        raise ValueError("model must be a dictionary")
    if model.get("model_version") != MODEL_VERSION:
        raise ValueError("Unsupported model_version; use %s and its dedicated interface" % MODEL_VERSION)
    if model.get("row_order") != ROW_ORDER:
        raise ValueError("row_order must be exactly %r" % ROW_ORDER)
    H = _array(model.get("H_px_per_mm"), (4, 3), "H_px_per_mm")
    _array(model.get("offset_px"), (4,), "offset_px")
    if _matrix_rank_and_condition(H)[0] != 3:
        raise ValueError("H_px_per_mm must have rank 3 for 3-D reconstruction")
    for view, rows in (("left", H[:2]), ("right", H[2:])):
        if _matrix_rank_and_condition(rows)[0] != 2:
            raise ValueError("%s image block must have rank 2" % view)
    d = model.get("distortion_reference")
    if not isinstance(d, dict):
        raise ValueError("distortion_reference must be a dictionary")
    K = _array(d.get("K"), (2, 2), "distortion_reference.K")
    if _matrix_rank_and_condition(K)[0] != 2:
        raise ValueError("distortion_reference.K must be invertible")
    _array(d.get("c"), (2,), "distortion_reference.c")
    k = _array(d.get("kappa"), (2,), "distortion_reference.kappa")
    scale = _positive_scalar(d.get("q_scale"), "distortion_reference.q_scale")
    radius = _positive_scalar(d.get("radius_limit_mm"), "distortion_reference.radius_limit_mm")
    # d[r * (1 + k1*r^2/s^2 + k2*r^4/s^4)] / dr.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        top = float(np.square(np.float64(radius) / np.float64(scale)))
    if not np.isfinite(top):
        raise ValueError("radius_limit_mm/q_scale must define a finite normalized radial domain")
    candidates = [0.0, top]
    if k[1] > 0:
        stationary = -3.0 * k[0] / (10.0 * k[1])
        if 0.0 < stationary < top:
            candidates.append(float(stationary))
    with np.errstate(over="ignore", invalid="ignore"):
        slopes = 1.0 + 3.0 * k[0] * np.asarray(candidates) + 5.0 * k[1] * np.asarray(candidates) ** 2
        max_distorted_radius = radius * (1.0 + k[0] * top + k[1] * top ** 2)
    if not np.all(np.isfinite(slopes)) or np.min(slopes) <= 1e-12:
        raise ValueError("Radial map must be strictly monotone with positive derivative on the full radius domain")
    if not np.isfinite(max_distorted_radius) or max_distorted_radius <= 0:
        raise ValueError("Distorted radius boundary is not positive and finite")
    if "default_crop_origins_px" not in model:
        raise ValueError("default_crop_origins_px must be provided, or null if unknown")
    if model["default_crop_origins_px"] is not None:
        _crop_origins(model["default_crop_origins_px"])
    if not isinstance(model.get("coordinates"), (str, dict)) or not model["coordinates"]:
        raise ValueError("coordinates must document the effective camera frame, not a physical prism frame")
    # Three-term/Brown distortions require a different implementation/version;
    # accepting them while silently ignoring fields is not permissible.
    if any(key in d for key in ("tangential", "distortion_center_px", "p1", "p2", "k3")):
        raise ValueError("This model version supports only the specified fixed two-term radial map")


def load_model(path: Any) -> dict:
    """Load and validate only this candidate schema, never a legacy calibration."""
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        model = json.load(handle)
    validate_model(model)
    return model


def _points(value: Any, columns: int, name: str, finite: bool = True) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a numeric N x %d array" % (name, columns)) from exc
    if points.ndim != 2 or points.shape[1] != columns:
        raise ValueError("%s must have shape (N, %d), got %s" % (name, columns, points.shape))
    if finite and not np.all(np.isfinite(points)):
        raise ValueError("%s must contain only finite numbers" % name)
    return points


def _distortion_parts(model: dict) -> tuple:
    d = model["distortion_reference"]
    return (np.asarray(d["K"], float), np.asarray(d["c"], float),
            np.asarray(d["kappa"], float), float(d["q_scale"]), float(d["radius_limit_mm"]))


def _distort_pixels(ideal: np.ndarray, model: dict) -> tuple:
    K, c, k, scale, radius = _distortion_parts(model)
    result = np.full(ideal.shape, np.nan)
    q = np.linalg.solve(K, (ideal - c).T).T
    r = np.linalg.norm(q, axis=1)
    valid = np.all(np.isfinite(ideal), axis=1) & np.isfinite(r) & (r <= radius * (1.0 + 1e-12))
    if np.any(valid):
        r2 = np.sum(q[valid] ** 2, axis=1, keepdims=True) / scale ** 2
        result[valid] = (q[valid] * (1.0 + k[0] * r2 + k[1] * r2 ** 2)) @ K.T + c
    return result, valid


def _undistort_pixels(raw: np.ndarray, model: dict) -> tuple:
    K, c, k, scale, radius = _distortion_parts(model)
    result = np.full(raw.shape, np.nan)
    qd = np.linalg.solve(K, (raw - c).T).T
    rd = np.linalg.norm(qd, axis=1)
    top = (radius / scale) ** 2
    rd_max = radius * (1.0 + k[0] * top + k[1] * top ** 2)
    valid = np.all(np.isfinite(raw), axis=1) & np.isfinite(rd) & (rd <= rd_max * (1.0 + 1e-12))
    if np.any(valid):
        target = np.minimum(rd[valid], rd_max)
        lower = np.zeros_like(target)
        upper = np.full_like(target, radius)
        # A monotone scalar inverse, vectorised over rows. Bisection is bounded
        # and never jumps to another branch of the radial polynomial.
        for _ in range(70):
            mid = (lower + upper) * 0.5
            r2 = (mid / scale) ** 2
            mapped = mid * (1.0 + k[0] * r2 + k[1] * r2 ** 2)
            low_side = mapped < target
            lower = np.where(low_side, mid, lower)
            upper = np.where(low_side, upper, mid)
        r = (lower + upper) * 0.5
        factor = np.divide(r, target, out=np.ones_like(r), where=target > 0)
        result[valid] = (qd[valid] * factor[:, None]) @ K.T + c
    return result, valid


def distort_pixels(undistorted_px: Any, model: dict) -> np.ndarray:
    """N x 2 undistorted full-sensor pixels -> raw pixels; strict domain check."""
    validate_model(model)
    pixels = _points(undistorted_px, 2, "undistorted_px")
    raw, valid = _distort_pixels(pixels, model)
    if not np.all(valid):
        raise ValueError("Undistorted pixels outside radius_limit_mm at rows %r" % np.flatnonzero(~valid).tolist())
    return raw


def undistort_pixels(raw_px: Any, model: dict) -> np.ndarray:
    """N x 2 raw full-sensor pixels -> undistorted pixels; strict domain check."""
    validate_model(model)
    pixels = _points(raw_px, 2, "raw_px")
    ideal, valid = _undistort_pixels(pixels, model)
    if not np.all(valid):
        raise ValueError("Raw pixels outside invertible radial domain at rows %r" % np.flatnonzero(~valid).tolist())
    return ideal


def project(X_mm: Any, model: dict) -> np.ndarray:
    """Project N x 3 effective camera coordinates to N x 4 raw full pixels.

    Row order is fixed: [left.u, left.v, right.u, right.v]. Projection outside
    the calibrated radial domain raises ValueError rather than extrapolating.
    """
    validate_model(model)
    X = _points(X_mm, 3, "X_mm")
    ideal = X @ np.asarray(model["H_px_per_mm"], float).T + np.asarray(model["offset_px"], float)
    return np.column_stack([distort_pixels(ideal[:, :2], model), distort_pixels(ideal[:, 2:], model)])


def reconstruct(left_px: Any, right_px: Any, model: dict,
                coordinate_space: str = "full_frame",
                crop_origins_px: Optional[dict] = None) -> dict:
    """Triangulate matched image points without dropping any input row.

    ``coordinate_space='cropped'`` restores each branch's full-sensor origin
    BEFORE undistortion. Default origins are model metadata; an explicit dict
    can override them. The only other accepted space is ``'full_frame'``.

    NaN/Inf observations and out-of-domain rows yield valid_mask=False and
    X_mm=NaN. Errors are predicted minus observed raw pixels, shape (N, 2, 2),
    axes [row, left/right, u/v]. They are not whitened or filtered. ``condition``
    is the 2-norm condition number of H, not an accuracy or ray-validity test.
    The solve minimises unweighted UNDISTORTED pixel errors, not raw-pixel
    likelihood; this distinction matters when comparing residual statistics.
    """
    validate_model(model)
    left = _points(left_px, 2, "left_px", finite=False)
    right = _points(right_px, 2, "right_px", finite=False)
    if len(left) != len(right):
        raise ValueError("left_px and right_px must have the same number of rows")
    if coordinate_space not in ("full_frame", "cropped"):
        raise ValueError("coordinate_space must be 'full_frame' or 'cropped'")
    if coordinate_space == "full_frame" and crop_origins_px is not None:
        raise ValueError("crop_origins_px is only valid with coordinate_space='cropped'")
    left, right = left.copy(), right.copy()
    if coordinate_space == "cropped":
        chosen_origins = model["default_crop_origins_px"] if crop_origins_px is None else crop_origins_px
        if chosen_origins is None:
            raise ValueError("No default crop origins are known; provide explicit crop_origins_px")
        origins = _crop_origins(chosen_origins)
        left += origins["left"]
        right += origins["right"]
    n = len(left)
    reasons = [[] for _ in range(n)]
    ideal_parts = []
    branch_valid = []
    for view, pixels in (("left", left), ("right", right)):
        finite = np.all(np.isfinite(pixels), axis=1)
        safe_pixels = pixels.copy()
        safe_pixels[~finite] = np.nan
        ideal, valid = _undistort_pixels(safe_pixels, model)
        for i in np.flatnonzero(~finite):
            reasons[int(i)].append(view + "_nonfinite_pixels")
        for i in np.flatnonzero(finite & ~valid):
            reasons[int(i)].append(view + "_outside_radial_domain")
        ideal_parts.append(ideal)
        branch_valid.append(valid)
    valid = branch_valid[0] & branch_valid[1]
    H = np.asarray(model["H_px_per_mm"], float)
    offset = np.asarray(model["offset_px"], float)
    X = np.full((n, 3), np.nan)
    errors = np.full((n, 2, 2), np.nan)
    indices = np.flatnonzero(valid)
    if len(indices):
        ideal4 = np.column_stack(ideal_parts)[valid]
        # Match the rank threshold used by validate_model to avoid silently
        # truncating a matrix which passed validation.
        solved = (ideal4 - offset) @ np.linalg.pinv(H, rcond=max(H.shape) * np.finfo(float).eps).T
        reproj_ideal = solved @ H.T + offset
        pred_parts = []
        final_valid = np.ones(len(indices), dtype=bool)
        for view, start in (("left", 0), ("right", 2)):
            pred, domain_ok = _distort_pixels(reproj_ideal[:, start:start + 2], model)
            for j in np.flatnonzero(~domain_ok):
                reasons[int(indices[j])].append(view + "_reprojection_outside_radial_domain")
            final_valid &= domain_ok
            pred_parts.append(pred)
        keep = indices[final_valid]
        X[keep] = solved[final_valid]
        raw4 = np.column_stack([left, right])
        pred4 = np.column_stack(pred_parts)
        errors[keep] = (pred4[final_valid] - raw4[keep]).reshape(-1, 2, 2)
        valid[indices[~final_valid]] = False
    return {
        "X_mm": X,
        "valid_mask": valid,
        "raw_reprojection_errors_px": errors,
        "condition": _matrix_rank_and_condition(H)[1],
        "invalid_reasons": reasons,
        "coordinate_warning": COORDINATE_WARNING,
        "model_version": MODEL_VERSION,
    }
