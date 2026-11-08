"""Independent synthetic fitter tests; no claims about external data accuracy."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from affine_camera import MODEL_VERSION, ROW_ORDER, project, reconstruct
from calibrate_affine import board_points, fit_shared_affine, ideal_pixels


def synthetic_scene():
    """Twelve complete 6 x 9 grids with independent 3-axis rotations/translations."""
    rng = np.random.default_rng(1474)
    xyz, center = board_points(dict(rows=6, cols=9, pitch_mm=2.0))
    H = np.array([[-30.0, 0.0, 3.0], [0.5, -34.0, 0.1],
                  [-30.0, 0.0, -3.0], [0.5, -34.0, -0.1]])
    offset = np.array([820.0, 1200.0, 2470.0, 1220.0])
    rotations = Rotation.from_rotvec(rng.normal(0.0, 0.45, (12, 3))).as_matrix()
    translations = rng.normal(0.0, 3.0, (12, 3))
    camera_xyz = np.einsum('nij,kj->nki', rotations, xyz) + translations[:, None, :]
    model = dict(
        model_version=MODEL_VERSION, H_px_per_mm=H.tolist(), offset_px=offset.tolist(),
        row_order=ROW_ORDER.copy(),
        distortion_reference=dict(K=[[-30.0, 0.0], [0.5, -34.0]], c=[1687.5, 1351.5],
                                  kappa=[-0.025, 0.003], q_scale=60.0, radius_limit_mm=120.0),
        default_crop_origins_px=dict(left=[0, 0], right=[1688, 0]),
        coordinates=dict(system='effective_camera', unit='mm', note='Synthetic effective frame'),
    )
    ideal = camera_xyz @ H.T + offset
    return dict(rng=rng, xyz=xyz, center=center, H=H, offset=offset, rotations=rotations,
                translations=translations, X=camera_xyz, ideal=ideal, model=model)


class SharedAffineFitTests(unittest.TestCase):
    def test_board_center_pitch_and_id_order(self):
        xyz, center = board_points(dict(rows=6, cols=9, pitch_mm=3.0))
        self.assertEqual(xyz.shape, (54, 3))
        np.testing.assert_allclose(center, [12.0, 7.5, 0.0], atol=1e-14)
        np.testing.assert_allclose(xyz.mean(axis=0), 0.0, atol=1e-14)
        np.testing.assert_allclose(xyz[1] - xyz[0], [3, 0, 0], atol=1e-14)
        np.testing.assert_allclose(xyz[9] - xyz[0], [0, 3, 0], atol=1e-14)
        np.testing.assert_allclose(xyz[0] + center, [0, 0, 0], atol=1e-14)
        np.testing.assert_allclose(xyz[-1] + center, [24, 15, 0], atol=1e-14)

    def test_board_rejects_unsupported_or_invalid_geometry(self):
        for board in [dict(rows=1, cols=9, pitch_mm=3), dict(rows=6, cols=1, pitch_mm=3),
                      dict(rows=6.5, cols=9, pitch_mm=3), dict(rows=6, cols=9.1, pitch_mm=3),
                      dict(rows=True, cols=9, pitch_mm=3), dict(rows=float('inf'), cols=9, pitch_mm=3),
                      dict(rows=6, cols=9, pitch_mm=0), dict(rows=6, cols=9, pitch_mm=-3),
                      dict(rows=6, cols=9, pitch_mm=float('nan')),
                      dict(rows=6, cols=9, pitch_mm=3, points_csv='nonregular.csv')]:
            with self.subTest(board=board), self.assertRaises(ValueError):
                board_points(board)

    def test_twelve_pose_noiseless_ideal_raw_and_metric_recovery(self):
        scene = synthetic_scene()
        raw = np.array([project(X, scene['model']) for X in scene['X']])
        undistorted = np.array([ideal_pixels(p, scene['model']) for p in raw])
        np.testing.assert_allclose(undistorted, scene['ideal'], atol=2e-10, rtol=0)
        H, b, R, t, info = fit_shared_affine(undistorted, scene['xyz'])
        fitted_X = np.einsum('nij,kj->nki', R, scene['xyz']) + t[:, None, :]
        fitted_ideal = fitted_X @ H.T + b
        np.testing.assert_allclose(fitted_ideal, scene['ideal'], atol=1e-8, rtol=0)
        candidate = copy.deepcopy(scene['model'])
        candidate.update(H_px_per_mm=H.tolist(), offset_px=b.tolist())
        fitted_raw = np.array([project(X, candidate) for X in fitted_X])
        np.testing.assert_allclose(fitted_raw, raw, atol=1e-8, rtol=0)
        recovered = np.array([reconstruct(p[:, :2], p[:, 2:], candidate)['X_mm'] for p in raw])
        np.testing.assert_allclose(recovered, fitted_X, atol=1e-8, rtol=0)
        # Compare invariant lengths, never absolute axes: a global rigid gauge
        # and a possible planar-target mirror convention are not physical data.
        for X in recovered:
            np.testing.assert_allclose(pdist(X), pdist(scene['xyz']), atol=1e-8, rtol=0)
        # Cross-frame distances test that all poses share one common metric
        # camera, not independent arbitrary affine fits of each frame.
        np.testing.assert_allclose(pdist(recovered[:, ::7].reshape(-1, 3)),
                                   pdist(scene['X'][:, ::7].reshape(-1, 3)), atol=1e-8, rtol=0)
        np.testing.assert_allclose(H @ H.T, scene['H'] @ scene['H'].T, atol=1e-7, rtol=0)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)
        np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.tile(np.eye(3), (12, 1, 1)), atol=1e-12)
        self.assertTrue(info['success'])
        self.assertFalse(info['global_optimum_certified'])

    def test_noisy_compressed_objective_equals_full_point_l2(self):
        scene = synthetic_scene()
        noisy = scene['ideal'] + scene['rng'].normal(0.0, 0.05, scene['ideal'].shape)
        H, b, R, t, info = fit_shared_affine(noisy, scene['xyz'])
        xyz = scene['xyz']
        prediction = (np.einsum('nij,kj->nki', R, xyz) + t[:, None, :]) @ H.T + b
        direct = np.sum((prediction - noisy) ** 2) / len(xyz)
        # Independently rebuild OLS sufficient statistics and the orthogonal
        # translation residual, without reusing the fitter's private residual.
        design = np.column_stack([xyz[:, :2], np.ones(len(xyz))])
        coeff = np.array([np.linalg.lstsq(design, p, rcond=None)[0] for p in noisy])
        slope = coeff[:, :2, :].transpose(0, 2, 1)
        center = coeff[:, 2, :]
        null = np.linalg.svd(H.T, full_matrices=True)[2][-1]
        weights = np.sqrt(np.mean(xyz[:, :2] ** 2, axis=0))
        compressed = (np.sum(((np.einsum('ab,nbc->nac', H, R[:, :, :2]) - slope) * weights) ** 2)
                      + np.sum(((center - b) @ null) ** 2))
        nonaffine = sum(np.sum((design @ cf - p) ** 2) for cf, p in zip(coeff, noisy)) / len(xyz)
        self.assertGreater(nonaffine, 1e-4)
        self.assertGreater(compressed, 1e-5)
        self.assertAlmostEqual(direct, compressed + nonaffine, places=9)
        self.assertAlmostEqual(direct, info['full_point_sum_squares_per_grid_point'], places=10)
        self.assertAlmostEqual(compressed, info['final_compressed_sum_squares'], places=9)
        self.assertAlmostEqual(nonaffine, info['fixed_nonaffine_sum_squares_per_grid_point'], places=10)
        self.assertLess(info['final_compressed_sum_squares'], info['initial_compressed_sum_squares'])
        self.assertFalse(info['global_optimum_certified'])

    def test_repeated_parallel_plane_poses_rejected(self):
        scene = synthetic_scene()
        # Translation and in-plane rotation do not add out-of-plane normals.
        R = Rotation.from_euler('z', np.linspace(-30, 30, 12), degrees=True).as_matrix()
        X = np.einsum('nij,kj->nki', R, scene['xyz']) + scene['translations'][:, None, :]
        pixels = X @ scene['H'].T + scene['offset']
        with self.assertRaisesRegex(ValueError, 'pose diversity'):
            fit_shared_affine(pixels, scene['xyz'])

    def test_two_repeated_normals_fail_metric_upgrade(self):
        scene = synthetic_scene()
        # This has rank-3 slope data but only two distinct plane normals; it
        # exercises the separate rank-6 metric-upgrade identifiability guard.
        R = np.stack([np.eye(3), Rotation.from_euler('x', 30, degrees=True).as_matrix()] * 6)
        X = np.einsum('nij,kj->nki', R, scene['xyz']) + scene['translations'][:, None, :]
        pixels = X @ scene['H'].T + scene['offset']
        with self.assertRaisesRegex(ValueError, 'metric-upgrade'):
            fit_shared_affine(pixels, scene['xyz'])

    def test_rejects_insufficient_frames_and_uncentered_board(self):
        scene = synthetic_scene()
        with self.assertRaisesRegex(ValueError, '>=4'):
            fit_shared_affine(scene['ideal'][:3], scene['xyz'])
        with self.assertRaisesRegex(ValueError, 'centered'):
            fit_shared_affine(scene['ideal'], scene['xyz'] + [1, 0, 0])
        bad = scene['ideal'].copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Finite'):
            fit_shared_affine(bad, scene['xyz'])


if __name__ == '__main__':
    unittest.main()
