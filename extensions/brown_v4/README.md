# Optional Brown v4 research source

This isolated snapshot implements `bbs-transverse-bundle-brown-v4`, the extended
physical model used for the calibration case A. The 18 Python
source files are listed with hashes in SOURCE_MANIFEST.json.

The model has 14 global parameters: three entries of the shared effective pixel
matrix, three prism rotation-vector components, three radial coefficients, one
transverse prism offset, two tangential coefficients, and two coordinates of a
separate distortion center. Each training board pose adds six local parameters.
Forward projection includes the shared transverse offset before centered Brown
distortion. Reconstruction uses the matching Brown inverse and least squares in
the undistorted reference-plane coordinates.

Keep this folder separate from the base-model release: both contain a package
named `bbs_calib` and their model adapters are not interchangeable. Run commands
with this folder as the working directory. The legacy `reconstruct_points`
adapter deliberately rejects translated/Brown models.

## Installation and entry points

The snapshot was checked with Python 3.8.20, NumPy 1.24.4, SciPy 1.10.1 and OpenCV
4.10.0. Install the compatible dependencies in requirements.txt in an isolated
environment. OpenCV is imported by the shared I/O module even for point-only use.

```text
python -B joint_calibrate.py --help
python -B joint_reconstruct.py --help
```

Calibration requires an extraction directory with a reviewed `config.json` and
`observations.csv`, a valid initial baseline calibration, and a calibration file
that supplies the fixed training/held-out frame split. This source snapshot does
not include an image feature extractor. A typical command, with user-supplied
relative paths, is:

```text
python -B joint_calibrate.py --source inputs/extract --initial-calibration inputs/baseline.json --split-from inputs/baseline.json --output results/brown_v4 --starts 3 --max-nfev 400 --workers 4 --offset-bound-mm 10 --path-weight 540 --radial-order 3 --tangential --free-distortion-center --distortion-center-bound-px 300 --tangential-bound 0.05 --confirm-order
```

For reconstruction, provide an Nx4 NumPy array whose columns are
`u_left, v_left, u_right, v_right`, and the matching accepted Brown calibration:

```text
python -B joint_reconstruct.py --calibration inputs/brown_calibration.json --matches inputs/matches.npy --pixel-coordinates full_frame --output results/xyz.npz
```

For crop-local coordinates, supply both actual crop origins using the documented
CLI options. Never add crop origins a second time to full-frame coordinates.
Both commands write new output locations and reject inappropriate model versions.

## Scope and validation

The optimizer uses robust pixel residuals and optical-path penalties. No board
distance, rigid-motion, or zero-strain objective was added to the reported Brown
fit. Low reprojection error does not certify the independent accuracy of prism
position, distortion center or sensor tilt; the reported global rank is 13 of 14
under the recorded numerical threshold. The physical aperture was not supplied.

VALIDATION.json records import/CLI checks and a read-only replay through this
copied source using the locally archived experiment. It reproduces the saved
held-out pixel RMS and free-reconstruction geometry without refitting.
The real images, observed coordinates, fitted calibration and fixed split files
used for that local validation are not included. This source package alone is
therefore not a complete reproduction archive for the real experiment.

## Release status

This research source is covered by the repository's BSD-3-Clause LICENSE.
Its citation metadata remain pending. Real data are
not included or newly published by this snapshot.
