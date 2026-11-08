# Scheimpflug bi-prism bi-telecentric stereo calibration

A bi-prism places two oblique views on one sensor. With a bi-telecentric lens
and a tilted sensor, this makes a compact stereo imaging system, but its two
pixel maps depend jointly on refraction and the Scheimpflug geometry. This
repository fits those maps from a calibration target and reconstructs 3D
points from image pairs whose correspondences are already known.

The code accompanies *Joint calibration and identifiability of a Scheimpflug
bi-prism bi-telecentric stereo system*. Stereo PIV uses two oblique views of
a laser sheet to recover three velocity components from particle-image
displacements; calibration supplies the viewing geometry. Stereo DIC uses
matched speckles to measure 3D surface shape and displacement. This
repository handles calibration and point reconstruction; it does not
correlate images or calculate velocity, displacement, or strain fields.

## Start with a synthetic calibration

Use Python 3.10 in a fresh environment. From the repository root:

```bash
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps .
python calibrate.py demo --output work/synthetic_demo
```

The demo renders a dot grid, detects its centers, fits the physical model,
and evaluates image points held out from fitting. It writes the generated
images and calibration reports under the ignored `work/synthetic_demo/`
directory. Output directories must be new or empty.

In the [recorded run](results/synthetic_demo/summary.json), all 480
synthetic centers were found. Euclidean reprojection RMS was 0.0526 px on
training points and 0.0674 px on held-out evaluation points. These numbers
check the implementation on generated images; they are not measurements of
a physical camera or a PIV/DIC system.

## Which code belongs to which model?

| Path | Role |
| --- | --- |
| [`bbs_calib/`](bbs_calib/) | Main package: bi-prism ray tracing, an effective tilted-sensor pixel map, two-term radial distortion, calibration, validation, and 3D point reconstruction. |
| [`extensions/brown_v4/`](extensions/brown_v4/) | Extended physical model used for Case A in the manuscript: transverse prism offset and Brown distortion. It is an isolated source snapshot with its own commands and dependencies. |
| [`candidate_affine/`](candidate_affine/) | Shared affine stereo comparison. Its fitted matrix is not decomposed into mechanical prism or sensor angles. |
| [`experiments/`](experiments/) | Synthetic displacement-noise and ray-geometry checks. |

The root model uses a shared effective pixel map after the two prism ray
paths. Its projection, coordinates, and assumptions are documented in
[`docs/model.md`](docs/model.md). The Brown extension uses a different
calibration schema; its files cannot be loaded as root-model calibrations.
See its [README](extensions/brown_v4/README.md) for the exact inputs and
entry points.

## Use with measured observations

Start by recording the board geometry, prism dimensions, refractive index,
pixel size, crop origins, and the mapping from image views to physical prism
branches. The configuration template leaves unknown hardware values blank:

```bash
python calibrate.py init --template cropped --output work/hardware.json
```

The [usage guide](docs/public_usage.md) gives the extraction, center review,
fit, inspection, and reconstruction commands. Inspect point identities
before confirming their order. The fitted calibration can then reconstruct
matched image points, for example:

```bash
python calibrate.py reconstruct --calibration path/to/calibration.json --pairs path/to/matches.csv --output work/points3d
```

`matches.csv` needs `point_id`, `u_left_px`, `v_left_px`, `u_right_px`,
and `v_right_px`. This root-model command does not find correspondences.
Use the Brown extension's separate reconstruction command for Brown fits.

## Check the released code

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s candidate_affine/tests -v
python experiments/oracle_displacement.py --output work/oracle_displacement
```

The oracle experiment starts from a known calibration and exact
correspondences, then compares predicted displacement noise with Monte
Carlo samples. Its recorded CSV and JSON outputs are in
[`results/oracle_displacement/`](results/oracle_displacement/).
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) records the tested
environment, commands, numerical results, and what each check establishes.
GitHub Actions runs the tests on Windows and Ubuntu.

## Data and interpretation

No real images, measured image-point observations, or fitted real-data
calibrations are distributed here. The Brown extension includes a
read-only [validation record](extensions/brown_v4/VALIDATION.json) from
locally archived observations, but those observations are required to
rerun the real fit. The available synthetic results test algorithms and
geometry, not instrument accuracy.

A low reprojection residual alone does not establish mechanical parameter
accuracy or an accurate velocity or displacement field. Review valid ray
paths, held-out errors, parameter identifiability, and independent
measurements before using a fit quantitatively. Further limits are in
[`docs/limitations.md`](docs/limitations.md).

The source is released under [BSD-3-Clause](LICENSE). That license covers
this repository's code, not the separate manuscript or undistributed
experimental data.
