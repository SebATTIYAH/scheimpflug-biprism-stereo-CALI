# Effective affine comparison

This separate comparison fits one shared 4 x 3 metric projection matrix to
multiple poses of a planar grid. It retains the fixed two-term radial correction
of a supplied baseline. The matrix is not decomposed into mechanical prism or
sensor-tilt parameters. Its coordinates have a global rigid-frame and planar
reflection convention; axis signs are not a physical depth-direction measurement.

Run from the repository root:

```bash
python -m unittest discover -s candidate_affine/tests -v
python candidate_affine/calibrate_affine.py fit --baseline path/to/baseline_fit --output work/affine_fit
python candidate_affine/calibrate_affine.py reconstruct --model work/affine_fit/calibration_candidate.json --points matched_points.json --output reconstructed.json
```

The optional fit command requires reviewed baseline `calibration.json`,
`observations.csv`, `validation_observations.csv`, and `validation.json`.
Real experiment data are not distributed. The 18 synthetic tests run without
them, including noiseless metric recovery, compressed-objective equivalence,
pose-diversity rejection, inverse-domain checks, and crop-origin handling.

The output schema `shared_affine_stereo_v1_candidate` is intentionally separate
from `bbs-shared-ray-radial-v1`; do not rename or interchange calibration files.
`valid_mask` denotes inverse-domain validity, not Snell ray-path certification.
The fitter minimizes undistorted pixel L2 error with no zero-strain penalty.
For held-out frames, only the original half-point subset determines pose;
the complementary points determine the reported raw pixel residuals.

This directory contains repository scripts and is included in the source
archive. Only `bbs_calib` is installed as a library by the Python wheel.
