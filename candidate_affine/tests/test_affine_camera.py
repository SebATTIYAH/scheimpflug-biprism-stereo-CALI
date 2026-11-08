"""Synthetic contract tests; not evidence of real-world metrological accuracy."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from affine_camera import (MODEL_VERSION, ROW_ORDER, distort_pixels, load_model,
                           project, reconstruct, undistort_pixels, validate_model)


def example_model():
    return {
        "model_version": MODEL_VERSION,
        "H_px_per_mm": [[-30.0, 0.0, 3.0], [0.5, -34.0, 0.1],
                        [-30.0, 0.0, -3.0], [0.5, -34.0, -0.1]],
        "offset_px": [820.0, 1200.0, 2470.0, 1220.0],
        "row_order": ROW_ORDER.copy(),
        "distortion_reference": {
            "K": [[-30.0, 0.0], [0.5, -34.0]], "c": [1687.5, 1351.5],
            "kappa": [-0.025, 0.003], "q_scale": 60.0,
            "radius_limit_mm": 120.0,
        },
        "default_crop_origins_px": {"left": [0, 0], "right": [1688, 0]},
        "coordinates": {"system": "effective_camera", "unit": "mm",
                        "note": "Not the original physical prism frame"},
    }


class CandidateCameraTests(unittest.TestCase):
    def setUp(self):
        self.model = example_model()
        self.X = np.random.default_rng(4321).uniform(-8, 8, size=(100, 3))

    def test_synthetic_roundtrip(self):
        pixels = project(self.X, self.model)
        result = reconstruct(pixels[:, :2], pixels[:, 2:], self.model)
        np.testing.assert_allclose(result["X_mm"], self.X, atol=1e-10)
        np.testing.assert_allclose(result["raw_reprojection_errors_px"], 0, atol=1e-10)
        self.assertTrue(result["valid_mask"].all())
        self.assertEqual(result["invalid_reasons"], [[] for _ in self.X])
        self.assertGreater(result["condition"], 1.0)

    def test_full_frame_equals_cropped_default_and_override(self):
        p = project(self.X, self.model)
        default = reconstruct(p[:, :2], p[:, 2:] - [1688, 0], self.model, "cropped")
        custom = reconstruct(p[:, :2] - [11, 4], p[:, 2:] - [1710, 9], self.model,
                             "cropped", {"left": [11, 4], "right": [1710, 9]})
        np.testing.assert_allclose(default["X_mm"], self.X, atol=1e-10)
        np.testing.assert_allclose(custom["X_mm"], self.X, atol=1e-10)

    def test_unknown_crop_origins_require_explicit_choice(self):
        self.model["default_crop_origins_px"] = None
        p = project(self.X, self.model)
        full = reconstruct(p[:, :2], p[:, 2:], self.model)
        np.testing.assert_allclose(full["X_mm"], self.X, atol=1e-10)
        with self.assertRaisesRegex(ValueError, "No default crop origins"):
            reconstruct(p[:, :2], p[:, 2:] - [1688, 0], self.model, "cropped")
        crop = reconstruct(p[:, :2], p[:, 2:] - [1688, 0], self.model, "cropped",
                           {"left": [0, 0], "right": [1688, 0]})
        np.testing.assert_allclose(crop["X_mm"], self.X, atol=1e-10)

    def test_radial_helpers_and_exact_zero_distortion(self):
        ideal = self.X[:, :2] * 10 + [1000, 1200]
        np.testing.assert_allclose(undistort_pixels(distort_pixels(ideal, self.model), self.model), ideal, atol=1e-10)
        self.model["distortion_reference"]["kappa"] = [0, 0]
        expected = self.X @ np.asarray(self.model["H_px_per_mm"]).T + self.model["offset_px"]
        np.testing.assert_allclose(project(self.X, self.model), expected, atol=1e-10)

    def test_invalid_rows_not_silently_removed(self):
        p = project(self.X[:5], self.model)
        p[1, 0] = np.nan
        p[2, 3] = np.inf
        p[3, 0] = 1e8
        with np.errstate(invalid="ignore"):
            result = reconstruct(p[:, :2], p[:, 2:], self.model)
        np.testing.assert_array_equal(result["valid_mask"], [True, False, False, False, True])
        self.assertEqual(result["X_mm"].shape, (5, 3))
        self.assertEqual(result["raw_reprojection_errors_px"].shape, (5, 2, 2))
        self.assertTrue(np.isnan(result["X_mm"][1:4]).all())
        self.assertIn("left_nonfinite_pixels", result["invalid_reasons"][1])
        self.assertIn("right_nonfinite_pixels", result["invalid_reasons"][2])
        self.assertIn("left_outside_radial_domain", result["invalid_reasons"][3])

    def test_radius_boundary_and_outside(self):
        d = self.model["distortion_reference"]
        K, c, limit = np.asarray(d["K"]), np.asarray(d["c"]), d["radius_limit_mm"]
        boundary = np.array([[limit, 0.0]]) @ K.T + c
        np.testing.assert_allclose(undistort_pixels(distort_pixels(boundary, self.model), self.model), boundary, atol=1e-9)
        with self.assertRaises(ValueError):
            distort_pixels(np.array([[limit * 1.001, 0]]) @ K.T + c, self.model)
        with self.assertRaises(ValueError):
            undistort_pixels([[1e8, 1e8]], self.model)
        with self.assertRaises(ValueError):
            project([[1e6, 0, 0]], self.model)

    def test_nonmonotone_radial_interior_is_rejected(self):
        # f'(s)=1-6s+5s^2 has positive endpoints at s=0 and s=4,
        # but a negative interior. A boundary-only check would miss this.
        self.model["distortion_reference"]["kappa"] = [-2.0, 1.0]
        with self.assertRaisesRegex(ValueError, "monotone"):
            validate_model(self.model)
        self.model["distortion_reference"]["kappa"] = [-0.5, 0]
        with self.assertRaisesRegex(ValueError, "monotone"):
            validate_model(self.model)

    def test_model_schema_and_shape_validation(self):
        edits = [
            ("H_px_per_mm", np.ones((4, 3)).tolist()),
            ("H_px_per_mm", np.eye(3).tolist()),
            ("offset_px", [1, 2, 3]),
            ("offset_px", [1, 2, 3, float("nan")]),
            ("row_order", ["right.u", "right.v", "left.u", "left.v"]),
            ("model_version", "legacy"),
        ]
        for key, value in edits:
            changed = copy.deepcopy(self.model)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_model(changed)
        for key, value in [("K", [[1, 0], [0, 0]]), ("kappa", [0, 0, 0]),
                           ("q_scale", 0), ("radius_limit_mm", -1),
                           ("tangential", [0, 0])]:
            changed = copy.deepcopy(self.model)
            changed["distortion_reference"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_model(changed)

    def test_reject_bad_input_dimensions_and_spaces(self):
        for bad in ([1, 2, 3], [[1, 2]], [[1, 2, float("nan")]]):
            with self.assertRaises(ValueError):
                project(bad, self.model)
        with self.assertRaises(ValueError):
            reconstruct([[0, 1]], [[0, 1], [2, 3]], self.model)
        with self.assertRaises(ValueError):
            reconstruct([0, 1], [0, 1], self.model)
        with self.assertRaises(ValueError):
            reconstruct([[0, 1]], [[0, 1]], self.model, "unknown")
        with self.assertRaises(ValueError):
            reconstruct([[0, 1]], [[0, 1]], self.model, crop_origins_px={"left": [0, 0], "right": [1, 0]})

    def test_load_model_and_explicit_candidate_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(self.model), encoding="utf-8")
            loaded = load_model(path)
            self.assertEqual(loaded["model_version"], MODEL_VERSION)
            # No physical `state`/view_branches prism schema masquerading as a
            # legacy calibration. Dedicated reader rejects the reverse mix-up.
            self.assertNotIn("state", loaded)
            self.assertNotIn("view_branches", loaded)
            path.write_text(json.dumps({"state": {}, "schema_version": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported model_version"):
                load_model(path)

    def test_empty_arrays_preserve_shape(self):
        p = project(np.empty((0, 3)), self.model)
        self.assertEqual(p.shape, (0, 4))
        result = reconstruct(p[:, :2], p[:, 2:], self.model)
        self.assertEqual(result["X_mm"].shape, (0, 3))
        self.assertEqual(result["invalid_reasons"], [])


if __name__ == "__main__":
    unittest.main()
