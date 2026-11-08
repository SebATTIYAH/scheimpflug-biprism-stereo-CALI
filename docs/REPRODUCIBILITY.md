# Reproduction record

These checks concern the root physical-core implementation and separate affine
comparison. The Brown extension under `extensions/brown_v4/` is an isolated
source snapshot with its own requirements and validation record. Git records
the curated source tree, and `tools/export_public.py` writes per-file SHA-256
hashes to its generated release manifest. The numerical model and solver are
unchanged.

Validation environment: Python 3.8.20, NumPy 1.24.4, SciPy 1.10.1,
OpenCV 4.10.0, Matplotlib 3.7.2. BLAS/OpenMP thread counts were set to one.
Commands were run from this source directory.

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s candidate_affine/tests -v
python -B calibrate.py demo --output work/synthetic_demo
python -B experiments/oracle_displacement.py --output work/oracle_displacement
```

The release copy reads JSON as UTF-8. This input-handling choice does not
change optical calculations. The final runs passed 156 physical-core tests and
18 affine-comparison tests. The selected source tests cover geometry,
reciprocity, ray tracing, reconstruction, distortion inversion, crop coordinates,
calibration, held-out partitions, observability checks, CLI, extraction and export.

The image demo found all 480 synthetic centers. Center RMS error against rendered
truth was 0.056300079 px. Training Euclidean reprojection RMS was 0.052582690 px
(360 image points); held-out RMS was 0.067418514 px (60 evaluation image points).
The 2 held-out frames use half their points for pose fitting and half for error
evaluation. Optimization convergence, ray geometry and local identifiability flags
were true; external validation was false. The `implementation_tested` field in
the legacy demo output remains its default false because that command does not
invoke the separate test suite; it is not silently changed to true here.

For the oracle-coordinate experiment, seed 314159, 12 conditions and 5000
displacement estimates per condition gave maximum independent scalar-ray/matrix
difference 1.5289e-14 mm, exact affine reduction at the recorded floating-point
precision, and a maximum Monte Carlo/predicted standard-deviation difference of
1.3056%. These are finite-sample numerical checks under fixed calibration.
The noise design and omissions are specified in `experiments/README.md` and the
machine-readable report. Do not treat them as measured apparatus accuracy.

The original affine real-board fit was also re-run locally and the measured
three-dimensional board diagnostics were independently re-evaluated. They match
the archived results. Those original datasets and fitted calibration files are
not distributed in this candidate; the companion manuscript evidence audit
records their provenance. The public package therefore reproduces synthetic
checks without access credentials, while real-board reproduction still requires
the reviewed original observations and calibration inputs.

The Python wheel was built locally with no dependency downloads, installed into
an isolated target directory, and checked by importing its bundled synthetic
configuration and running `doctor` outside the source root. This verifies the
installed core and package resources on the recorded environment.

The [GitHub Actions workflow](../.github/workflows/tests.yml) runs four jobs on Windows
and Ubuntu. The Python 3.10 jobs install the root package, run the core and
affine tests, generate a short oracle experiment, and check the source export
and Git tree. Separate jobs check both Brown
command-line entry points. These checks do not rerun the withheld measured-data
calibrations. The source repository uses BSD-3-Clause; citation authors and
publication identifiers remain deliberately unset.
