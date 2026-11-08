"""Standalone effective affine stereo candidate, deliberately not a production model.

Fit uses complete regular grids and a training-only fixed radial correction from
an existing physical calibration. No mechanical prism/Scheimpflug angle is fit.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

try:
    from .affine_camera import load_model, validate_model, project, reconstruct, undistort_pixels
except ImportError:
    from affine_camera import load_model, validate_model, project, reconstruct, undistort_pixels


def read_json(path):
    with open(path, encoding='utf-8-sig') as stream:
        return json.load(stream)


def save_json(path, value):
    # Exclusive creation: never overwrite an earlier result.
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nearest_rotation(a):
    u, _, vt = np.linalg.svd(a)
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


def metric_row(a, b):
    return [a[0]*b[0], a[1]*b[1], a[2]*b[2],
            a[0]*b[1]+a[1]*b[0], a[0]*b[2]+a[2]*b[0], a[1]*b[2]+a[2]*b[1]]


def metrics(error4):
    e = np.asarray(error4, dtype=float).reshape(-1, 2)
    lengths = np.linalg.norm(e, axis=1)
    return dict(count=len(e), euclidean_rms_px=float(np.sqrt(np.mean(lengths**2))),
                axis_rms_px=np.sqrt(np.mean(e**2, axis=0)).tolist(),
                median_px=float(np.median(lengths)), p95_px=float(np.percentile(lengths, 95)),
                max_px=float(lengths.max()))


def board_points(board):
    if any(isinstance(board[k], bool) or not np.isfinite(float(board[k])) or
           float(board[k]) != int(board[k]) for k in ('rows', 'cols')):
        raise ValueError('Board rows/cols must be finite integers, not booleans')
    rows, cols = int(board['rows']), int(board['cols'])
    pitch = float(board['pitch_mm'])
    if rows < 2 or cols < 2 or not np.isfinite(pitch) or pitch <= 0:
        raise ValueError('Board needs >=2 rows/columns and positive finite pitch_mm')
    if board.get('points_csv'):
        raise ValueError('This candidate currently requires a regular, complete grid')
    ids = np.arange(rows*cols)
    xyz = np.column_stack([ids % cols, ids // cols, np.zeros(len(ids))])*pitch
    center = xyz.mean(axis=0)
    return xyz-center, center


def read_observations(paths):
    groups = {}
    for path in paths:
        with open(path, newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                if row['accepted'].lower() != 'true':
                    continue
                if row['coordinate_space'] != 'full_frame':
                    raise ValueError('Fit requires full_frame observations; no crop origin is guessed')
                if row['view'] not in ('left', 'right'):
                    raise ValueError('Unsupported view: ' + row['view'])
                key = (row['view'], int(row['point_id']))
                value = [float(row['u_px']), float(row['v_px'])]
                entries = groups.setdefault(row['image_id'], {})
                if key in entries and entries[key] != value:
                    raise ValueError('Conflicting observation for ' + row['image_id'])
                entries[key] = value
    return groups


def frame_pixels(groups, frame, npoints):
    entries = groups[frame]
    expected = {(v, i) for v in ('left', 'right') for i in range(npoints)}
    if set(entries) != expected:
        raise ValueError('Complete paired grid required; missing/extra IDs in ' + frame)
    pixels = np.array([entries['left', i] + entries['right', i] for i in range(npoints)])
    if not np.isfinite(pixels).all():
        raise ValueError('Non-finite observation in ' + frame)
    return pixels


def ideal_pixels(raw4, model):
    return np.column_stack([undistort_pixels(raw4[:, :2], model),
                            undistort_pixels(raw4[:, 2:], model)])


def fit_shared_affine(ideal, xyz, max_nfev=400):
    """Train H, offset and rigid poses using complete centered regular grids.

    Variable-projection removes each frame's translation exactly. The compressed
    slope/center objective is full undistorted point L2 divided by N, less a
    parameter-independent non-affine term. This is NOT raw-pixel robust BA.
    """
    ideal = np.asarray(ideal, float)
    if ideal.ndim != 3 or ideal.shape[1:] != (len(xyz), 4) or len(ideal) < 4:
        raise ValueError('Need >=4 complete paired grid frames')
    if not np.isfinite(ideal).all() or not np.allclose(xyz.mean(0), 0, atol=1e-10):
        raise ValueError('Finite observations and centered board coordinates required')
    pc = xyz[:, :2]
    if not np.allclose(pc.T @ pc, np.diag(np.sum(pc**2, axis=0)), atol=1e-8):
        raise ValueError('Compressed fit requires orthogonal complete regular-grid axes')
    design = np.column_stack([pc, np.ones(len(pc))])
    coefficients = np.array([np.linalg.lstsq(design, a, rcond=None)[0] for a in ideal])
    slopes = coefficients[:, :2, :].transpose(0, 2, 1)
    centers = coefficients[:, 2, :]
    offset = centers.mean(0)
    center_delta = centers-offset
    weights = np.sqrt(np.mean(pc**2, axis=0))
    u, singular, vt = np.linalg.svd(np.concatenate(slopes, axis=1), full_matrices=False)
    if singular[2] <= singular[0]*1e-8:
        raise ValueError('Insufficient out-of-plane pose diversity: rank <3')
    a = u[:, :3]*np.sqrt(singular[:3])
    q = np.sqrt(singular[:3])[:, None]*vt[:3]
    equations, target = [], []
    for i in range(len(ideal)):
        q1, q2 = q[:, 2*i], q[:, 2*i+1]
        equations.extend([metric_row(q1, q1), metric_row(q2, q2), metric_row(q1, q2)])
        target.extend([1., 1., 0.])
    equations, target = np.array(equations), np.array(target)
    lm, _, rank, metric_singular = np.linalg.lstsq(equations, target, rcond=None)
    if rank < 6 or metric_singular[-1] <= metric_singular[0]*1e-10:
        raise ValueError('Planar poses do not constrain all metric-upgrade components')
    metric = np.array([[lm[0], lm[3], lm[4]], [lm[3], lm[1], lm[5]], [lm[4], lm[5], lm[2]]])
    eigen = np.linalg.eigvalsh(metric)
    if eigen[0] <= eigen[-1]*1e-10:
        raise ValueError('Metric upgrade is not positive definite; no eigenvalue clipping permitted')
    c = np.linalg.cholesky(metric).T
    h = a @ np.linalg.inv(c)
    rotations = []
    for i in range(len(ideal)):
        ri = c @ q[:, 2*i:2*i+2]
        rotations.append(nearest_rotation(np.column_stack([ri, np.cross(ri[:, 0], ri[:, 1])])))
    rotations = np.array(rotations)
    gauge, _ = np.linalg.qr(h[:3].T)
    if np.linalg.det(gauge) < 0:
        gauge[:, 2] *= -1
    h, rotations = h @ gauge, gauge.T @ rotations
    index = np.tril_indices(4, 0, 3)
    x0 = np.r_[h[index], Rotation.from_matrix(rotations).as_rotvec().ravel()]
    nf = len(ideal)

    def unpack(x):
        hh = np.zeros((4, 3))
        hh[index] = x[:9]
        rr = Rotation.from_rotvec(x[9:].reshape(-1, 3)).as_matrix()
        return hh, rr

    def residual(x):
        hh, rr = unpack(x)
        slopes_error = (np.einsum('ab,nbc->nac', hh, rr[:, :, :2])-slopes)*weights
        normal = np.linalg.svd(hh.T, full_matrices=True)[2][-1]
        return np.column_stack([slopes_error.reshape(nf, 8), center_delta @ normal]).ravel()

    sparsity = lil_matrix((nf*9, 9+nf*3), dtype=int)
    sparsity[:, :9] = 1
    for i in range(nf):
        sparsity[i*9:i*9+8, 9+i*3:12+i*3] = 1
    opt = least_squares(residual, x0, jac_sparsity=sparsity.tocsr(), x_scale='jac',
                        max_nfev=max_nfev, ftol=1e-11, gtol=1e-10, xtol=1e-11, tr_solver='lsmr')
    h, rotations = unpack(opt.x)
    # Fix signs by a proper rotation, not by a reflection or rescaling.
    signs = [1 if h[0, 0] >= 0 else -1, 1 if h[1, 1] >= 0 else -1]
    sign_gauge = np.diag(signs + [signs[0]*signs[1]])
    h, rotations = h @ sign_gauge, sign_gauge.T @ rotations
    translations = center_delta @ np.linalg.pinv(h).T
    pred = (np.einsum('nij,kj->nki', rotations, xyz)+translations[:, None, :]) @ h.T+offset
    nonaffine = np.array([design @ cf-pix for cf, pix in zip(coefficients, ideal)])
    compressed = float(np.sum(residual(opt.x)**2))
    direct = float(np.sum((pred-ideal)**2)/len(xyz))
    constant = float(np.sum(nonaffine**2)/len(xyz))
    if not np.isclose(direct, compressed+constant, rtol=1e-7, atol=1e-7):
        raise RuntimeError('Compressed objective does not match full-point L2')
    diagnostics = dict(success=bool(opt.success), status=int(opt.status), message=str(opt.message),
                       nfev=int(opt.nfev), optimality=float(opt.optimality), global_optimum_certified=False,
                       initial_compressed_sum_squares=float(np.sum(residual(x0)**2)),
                       final_compressed_sum_squares=compressed, full_point_sum_squares_per_grid_point=direct,
                       fixed_nonaffine_sum_squares_per_grid_point=constant,
                       slope_stack_singular_values=singular.tolist(),
                       metric_constraint_singular_values=metric_singular.tolist(),
                       metric_constraint_condition=float(metric_singular[0]/metric_singular[-1]),
                       metric_eigenvalues=eigen.tolist(), H_singular_values=np.linalg.svd(h, compute_uv=False).tolist())
    return h, offset, rotations, translations, diagnostics


def common_crop_origins(cal):
    origins = cal.get('crop_origins_by_image', {})
    values = list(origins.values())
    if values and all(v == values[0] for v in values):
        return values[0]
    default = cal.get('default_crop_origins')
    if isinstance(default, dict) and all(default.get(v) is not None for v in ('left', 'right')):
        return default
    return None  # Caller must explicitly provide crop origins when they vary.


def fit_command(args):
    base, output = Path(args.baseline).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError('Output must be a new directory (existing results are never overwritten)')
    cal_path, val_path = base/'calibration.json', base/'validation.json'
    cal, old_val = read_json(cal_path), read_json(val_path)
    if cal.get('model_version') != 'bbs-shared-ray-radial-v1':
        raise ValueError('Only the original bbs-shared-ray-radial-v1 fixed distortion is supported')
    observations = [base/'observations.csv']
    if (base/'validation_observations.csv').is_file():
        observations.append(base/'validation_observations.csv')
    groups = read_observations(observations)
    training = cal['training_frames']
    validation = [f for f in old_val['per_frame'] if f['status'] == 'evaluated']
    if len(training) != len(set(training)) or set(training) & {f['image_id'] for f in validation}:
        raise ValueError('Training/heldout split is not disjoint or contains duplicates')
    state, board = cal['state'], cal['config_snapshot']['board']
    xyz, board_center = board_points(board)
    model = dict(model_version='shared_affine_stereo_v1_candidate',
                 H_px_per_mm=[[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0]], offset_px=[0]*4,
                 row_order=['left.u', 'left.v', 'right.u', 'right.v'],
                 distortion_reference={k: state[k] for k in ('K', 'c', 'kappa', 'q_scale')},
                 coordinates=dict(system='effective_camera', unit='mm',
                                  note='Arbitrary proper rotation/translation gauge; planar-target mirror ambiguity remains. Not physical lens/prism coordinates.'),
                 status='candidate_only_not_production_calibration', board=board,
                 default_crop_origins_px=common_crop_origins(cal))
    model['distortion_reference']['radius_limit_mm'] = cal['radius_limit_mm']
    validate_model(model)
    raw = np.array([frame_pixels(groups, f, len(xyz)) for f in training])
    ideal = np.array([ideal_pixels(p, model) for p in raw])
    print('Training shared H on {} frames; all {} points/view retained'.format(len(raw), len(xyz)), flush=True)
    h, offset, rotations, translations, fit_info = fit_shared_affine(ideal, xyz, args.max_nfev)
    model.update(H_px_per_mm=h.tolist(), offset_px=offset.tolist(), training_frames=training,
                 validation_frames=[v['image_id'] for v in validation],
                 coordinates=dict(system='effective_camera', unit='mm',
                                  note='H lower-triangular gauge, positive H00/H11; origin is mean training board-center position. Z completes a right-handed frame. Mirror ambiguity unresolved; no mechanical-angle interpretation.'),
                 board_center_mm=board_center.tolist(),
                 pose_convention='X_camera = R @ (X_board - board_center_mm) + t_mm',
                 training_poses={f: dict(R=r.tolist(), t_mm=t.tolist()) for f, r, t in zip(training, rotations, translations)},
                 fit_diagnostics=fit_info,
                 fit_objective='Complete-grid undistorted pixel L2, fixed training-derived radial correction; variable-projection translation elimination. No strain penalty.',
                 source_files=[dict(path=str(p), sha256=digest(p)) for p in [cal_path, val_path]+observations],
                 limitations=['No independent rigid-motion/strain experiment is available.',
                              'Fixed radial correction is inherited from the training-only physical fit, not re-estimated.',
                              'No refractive ray-path or aperture certificate and no unique mechanical Scheimpflug/prism angles.',
                              'No global-optimality certificate; known pitch and pose diversity set the metric scale.',
                              'Current fitter requires complete regularly spaced grids; inverse reconstruction supports arbitrary matched points.'])
    validate_model(model)
    errors = []
    per_training = []
    for f, pixels, r, t in zip(training, raw, rotations, translations):
        e = project(xyz @ r.T+t, model)-pixels
        errors.extend(e.tolist())
        per_training.append(dict(image_id=f, **metrics(e)))
    report = dict(training=metrics(errors), training_per_frame=per_training, heldout_per_frame=[])
    held_errors = []
    design = np.column_stack([xyz[:, :2], np.ones(len(xyz))])
    for v in validation:
        f = v['image_id']
        pixels = frame_pixels(groups, f, len(xyz))
        ids = set(map(int, v['pose_fit_point_ids']))
        if not ids or not ids < set(range(len(xyz))):
            raise ValueError('Invalid heldout pose-fit IDs for ' + f)
        mask = np.array([i in ids for i in range(len(xyz))])
        expected_test = set(map(int, v['reprojection_test_point_ids']))
        if expected_test != set(np.flatnonzero(~mask)):
            raise ValueError('Original heldout test IDs are not exactly the disjoint complement for ' + f)
        coef = np.linalg.lstsq(design[mask], ideal_pixels(pixels, model)[mask], rcond=None)[0]
        rr = np.linalg.pinv(h) @ coef[:2].T
        rr = nearest_rotation(np.column_stack([rr, np.cross(rr[:, 0], rr[:, 1])]))
        initial = np.r_[Rotation.from_matrix(rr).as_rotvec(), np.linalg.pinv(h) @ (coef[2]-offset)]

        def pose_prediction(p):
            return project(xyz @ Rotation.from_rotvec(p[:3]).as_matrix().T+p[3:], model)

        pose = least_squares(lambda p: (pose_prediction(p)[mask]-pixels[mask]).ravel(), initial,
                             x_scale=np.r_[np.full(3, .1), np.ones(3)], max_nfev=250,
                             ftol=1e-11, xtol=1e-11, gtol=1e-10)
        e = (pose_prediction(pose.x)-pixels)[~mask]
        held_errors.extend(e.tolist())
        report['heldout_per_frame'].append(dict(image_id=f, pose_fit_point_ids=sorted(ids),
                                               test_point_ids=np.flatnonzero(~mask).tolist(),
                                               pose_success=bool(pose.success), pose_optimality=float(pose.optimality),
                                               metrics=metrics(e)))
    if not held_errors:
        raise ValueError('No heldout points to validate')
    report.update(heldout=metrics(held_errors),
                  note='Fixed training H/offset/distortion; six heldout pose DOF fit only original pose_fit_point_ids; disjoint IDs scored in raw full-frame pixels.')
    model['reprojection_metrics'] = dict(training=report['training'], heldout=report['heldout'])
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'calibration_candidate.json', model)
    save_json(output/'reprojection_validation.json', report)
    # Reload exactly what a downstream caller will read; verify raw forward/inverse.
    loaded = load_model(output/'calibration_candidate.json')
    x = xyz @ rotations[0].T+translations[0]
    p = project(x, loaded)
    back = reconstruct(p[:, :2], p[:, 2:], loaded)
    if not np.all(back['valid_mask']) or not np.allclose(x, back['X_mm'], rtol=0, atol=1e-7):
        raise RuntimeError('Serialized candidate forward/inverse check failed')
    save_json(output/'serialization_check.json', dict(max_roundtrip_error_mm=float(np.max(np.abs(x-back['X_mm']))), passed=True))
    print(json.dumps(dict(output=str(output), training=report['training'], heldout=report['heldout'], fit=fit_info), ensure_ascii=False, indent=2))


def reconstruct_command(args):
    model = load_model(args.model)
    data = read_json(args.points)
    origins = data.get('crop_origins_px')
    result = reconstruct(np.asarray(data['left_px'], float), np.asarray(data['right_px'], float), model,
                         coordinate_space=data.get('coordinate_space', 'full_frame'), crop_origins_px=origins)
    def clean(value):
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    save_json(Path(args.output), clean(result))
    print('Reconstructed {} / {} valid points'.format(int(np.sum(result['valid_mask'])), len(result['valid_mask'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit', help='New candidate fit; no production files are changed')
    fit.add_argument('--baseline', required=True, help='Old fit directory containing calibration/validation JSON and observations')
    fit.add_argument('--output', required=True, help='New non-existent output directory')
    fit.add_argument('--max-nfev', type=int, default=400)
    fit.set_defaults(function=fit_command)
    rec = sub.add_parser('reconstruct', help='Candidate-only JSON paired-point reconstruction')
    rec.add_argument('--model', required=True)
    rec.add_argument('--points', required=True)
    rec.add_argument('--output', required=True)
    rec.set_defaults(function=reconstruct_command)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
