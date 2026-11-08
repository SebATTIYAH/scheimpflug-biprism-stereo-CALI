from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Tuple
import numpy as np
from .io import read_json, read_csv


def load_config(path: Path) -> dict:
    c = read_json(path)
    root = Path(path).resolve().parent
    for section, key in [('images', 'directory'), ('board', 'points_csv'),
                         ('review', 'anchors_csv'), ('validation', 'translations_csv'),
                         ('images','crop_origins_csv')]:
        value = c.get(section, {}).get(key)
        if value:
            c[section][key] = str((root / value).resolve())
    if c.get('output'):
        c['output'] = str((root / c['output']).resolve())
    c['_config_path'] = str(Path(path).resolve())
    return c


def validate_config(c: dict, purpose: str = 'calibrate') -> None:
    from .image_inputs import input_mode, validate_origin
    mode=input_mode(c)
    auto_full=(mode=='full_frame' and c.get('full_frame_extraction',{}).get('detector')=='dual_grid_v1' and purpose=='calibrate')
    missing = []
    # Center-based calibration requires point spacing, not circle diameter.
    paths = []
    if not c.get('board', {}).get('points_csv'):
        paths += ['board.rows', 'board.cols', 'board.pitch_mm']
    if purpose in ('calibrate', 'detect'):
        paths += ['views.left.branch', 'views.right.branch']
        if mode=='full_frame' and not auto_full:
            paths += ['views.left.roi','views.right.roi']
    if purpose == 'detect':
        paths += ['images.directory', 'images.pattern']
    if purpose == 'calibrate':
        paths += ['prism.refractive_index', 'prism.wedge_L_deg', 'prism.wedge_R_deg',
                  'prism.thickness_mm', 'camera.c_px', 'camera.c_source',
                  'camera.omega_p_deg']
        if c.get('camera', {}).get('K_init') is None:
            paths += ['camera.m_px_per_mm']
    for path in paths:
        value = c
        for key in path.split('.'):
            value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            missing.append(path)
    if missing:
        raise ValueError('Missing required configuration (no hardware values are guessed): ' + ', '.join(missing))
    ids, P = board_points(c)
    if len(ids) < 4 or not np.isfinite(P).all() or len(set(ids)) != len(ids):
        raise ValueError('Board requires >=4 finite, uniquely numbered points')
    if np.linalg.matrix_rank(P-P.mean(axis=0)) < 2:
        raise ValueError('Board points must span at least a plane')
    diameter=c['board'].get('diameter_mm')
    if diameter is not None and (not np.isfinite(diameter) or diameter <= 0):
        raise ValueError('Circle diameter, when supplied, must be positive and finite')
    if purpose in ('calibrate', 'detect'):
        if {c['views'][v]['branch'] for v in ('left', 'right')} != {'L', 'R'}:
            raise ValueError('ROI views must map bijectively to physical branches L and R')
        for v in ('left', 'right'):
            if mode=='paired_crops' and c['views'][v].get('crop_origin_px') is not None:
                validate_origin(c['views'][v]['crop_origin_px'])
            r = c['views'][v].get('roi')
            if r is None and (mode=='paired_crops' or auto_full):
                continue
            if len(r) != 4 or min(r[:2]) < 0 or min(r[2:]) <= 0 or any(int(a) != a for a in r):
                raise ValueError('ROI must be integer [x,y,width,height]')
    if purpose == 'calibrate':
        p, cam = c['prism'], c['camera']
        for key in ('refractive_index','wedge_L_deg','wedge_R_deg','thickness_mm'):
            if not np.isscalar(p[key]) or not np.isfinite(p[key]):raise ValueError('Prism '+key+' must be finite')
        # Persist a canonical [x, y] pair for all downstream fitting/reporting.
        cam['pixel_size_um'] = pixel_sizes(cam.get('pixel_size_um'))
        for key,size in (('c_px',2),('omega_p_deg',3)):
            value=np.asarray(cam[key],float)
            if value.shape!=(size,) or not np.isfinite(value).all():raise ValueError('Invalid camera.'+key)
        for key in ('q_scale_mm', 'radius_limit_mm'):
            _optional_positive(cam, key)
        if p['refractive_index'] < 1 or p['thickness_mm'] <= 0:
            raise ValueError('Invalid prism refractive index or thickness')
        if any(not 0 <= p[k] < 70 for k in ('wedge_L_deg', 'wedge_R_deg')):
            raise ValueError('Wedge angles must be in [0,70) degrees')
        K = initial_K(c)
        if K.shape != (2, 2) or not np.isfinite(K).all() or K[0, 1] != 0 or K[0, 0]*K[1, 1] <= 0:
            raise ValueError('K_init must be lower triangular with same nonzero diagonal signs')
        opt = c.get('optimization', {})
        if opt.get('mode', 'effective') not in ('effective', 'physical_constrained'):
            raise ValueError('Unknown optimization mode')
        from .hardware_refinement import specifications
        specifications(c)


def board_points(c: dict) -> Tuple[list, np.ndarray]:
    b = c['board']
    if b.get('points_csv'):
        rows = read_csv(Path(b['points_csv']))
        return [str(r['point_id']) for r in rows], np.array([[float(r[k]) for k in ('X_mm','Y_mm','Z_mm')] for r in rows])
    rows, cols, pitch = b.get('rows'), b.get('cols'), b.get('pitch_mm')
    if rows is None or cols is None or pitch is None:
        raise ValueError('Missing board geometry')
    if not all(np.isscalar(x) and np.isfinite(x) for x in (rows,cols,pitch)) or rows < 2 or cols < 2 or int(rows) != rows or int(cols) != cols or pitch <= 0:
        raise ValueError('Grid requires integer rows/cols >=2 and positive pitch')
    rows,cols=int(rows),int(cols)
    return [str(i) for i in range(rows*cols)], np.array([[j*pitch, i*pitch, 0.] for i in range(rows) for j in range(cols)])


def pixel_sizes(value) -> list:
    """One positive size means square pixels; otherwise require explicit [x, y]."""
    number = lambda v: isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
    values = [value, value] if number(value) else value
    if (not isinstance(values, (list, tuple, np.ndarray)) or np.shape(values) != (2,) or
            not all(number(v) and np.isfinite(v) and v > 0 for v in values)):
        raise ValueError('camera.pixel_size_um requires a positive finite number (x=y), e.g. 3.69, '
                         'or two positive finite values [x, y], e.g. [3.69, 3.69]; unit: um')
    return [float(v) for v in values]


def initial_K(c: dict) -> np.ndarray:
    """Initial intrinsic matrix; independent of the radial normalization."""
    cam = c['camera']
    if cam.get('K_init') is not None:
        K = np.asarray(cam['K_init'], float)
    else:
        from .geometry import scheimpflug
        angles = np.radians(cam.get('tilt_initial_deg') or [2., 2.])
        raw_pixel = cam.get('pixel_size_um')
        pixel = pixel_sizes(raw_pixel) if raw_pixel is not None else None
        S = scheimpflug(*angles)
        # m_px_per_mm refers to u pixel pitch; known anisotropy must be explicit.
        factor = np.diag([1., pixel[0]/pixel[1]]) if pixel else np.eye(2)
        K = factor @ (cam['m_px_per_mm']*S)
    return K


def _optional_positive(cam: dict, key: str) -> None:
    value = cam.get(key)
    if value is not None and (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating)) or
            not np.isfinite(value) or value <= 0):
        raise ValueError('camera.'+key+' must be a positive finite number, or null for automatic selection')


def resolve_q_scale(c: dict, training_pixels: np.ndarray, training_frames: list) -> dict:
    """Resolve once, using full-frame TRAINING observations, and persist in config.

    This is a numerical convention, not a measured optical/hardware parameter.
    Existing numeric scales (including resolved snapshots) are never rescaled.
    """
    cam = c['camera']
    for key in ('q_scale_mm', 'radius_limit_mm'):
        _optional_positive(cam, key)
    if cam.get('q_scale_mm') is None:
        pixels = np.asarray(training_pixels, float)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) == 0 or not np.isfinite(pixels).all():
            raise ValueError('Automatic q_scale_mm requires finite, nonempty training pixel coordinates')
        K = initial_K(c)
        q = np.linalg.solve(K, (pixels-np.asarray(cam['c_px'], float)).T).T
        radius = np.linalg.norm(q, axis=1)
        scale = float(np.max(radius))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            raise ValueError('Cannot determine q_scale_mm from degenerate training radii; check initial K and principal point')
        info = dict(source='auto_training_initial_intrinsics', method='max norm(solve(K_initial, pixel_full_frame - c))',
                    q_scale_mm=scale, observation_count=len(pixels), training_frames=list(training_frames),
                    initial_K=K.tolist(), initial_c_px=list(cam['c_px']),
                    initial_radius_median_mm=float(np.median(radius)), initial_radius_max_mm=scale,
                    fixed_during_fit=True, hardware_measurement=False)
        cam['q_scale_mm'] = scale
    else:
        scale = float(cam['q_scale_mm'])
        previous = cam.get('q_scale_resolution')
        info = deepcopy(previous) if isinstance(previous, dict) and previous.get('q_scale_mm') == scale else dict(
            source='configured', q_scale_mm=scale, fixed_during_fit=True, hardware_measurement=False)
    if cam.get('radius_limit_mm') is None:
        cam['radius_limit_mm'] = 2*scale
        info['radius_limit_source'] = 'default_2_times_q_scale'
    else:
        info.setdefault('radius_limit_source', 'configured')
    info['radius_limit_mm'] = float(cam['radius_limit_mm'])
    cam['q_scale_resolution'] = info
    return info


def initial_state(c: dict) -> dict:
    cam = c['camera']
    _optional_positive(cam, 'q_scale_mm')
    if cam.get('q_scale_mm') is None:
        raise ValueError('Resolve automatic q_scale_mm from training observations before initial_state')
    return dict(K=initial_K(c), c=np.asarray(cam['c_px'], float),
                omega=np.radians(cam['omega_p_deg']), kappa=np.zeros(2),
                q_scale=float(cam['q_scale_mm']),
                radius_limit_mm=cam.get('radius_limit_mm') or 2*float(cam['q_scale_mm']),
                prism=deepcopy(c['prism']))
