# Reproducible numerical experiments

Run `python experiments/oracle_displacement.py --output work/oracle_new` from
the source root. The default seed is 314159 and each of 12 conditions uses
5000 displacement estimates. Reference outputs are in `results/oracle_displacement`.

Each experiment uses a known physical calibration, fixed exact correspondences,
and two independent noisy sets of image coordinates. It does not generate or
correlate particle images. Three prism geometries and four coordinate-noise
levels compare Monte Carlo standard deviations with the differential of the
implemented inverse. Calibration errors and systematic model errors are excluded.

If `M` stacks the two 2 x 3 ray matrices, `K` is the shared pixel map, and
`D'_s` is the radial-distortion Jacobian at one branch point, then
`J = pinv(M) blockdiag(inv(D'_R) inv(K), inv(D'_L) inv(K))`.
At independent times, `Cov(delta X) = sigma^2 (J0 J0^T + J1 J1^T)`.
This expression describes the implemented q-coordinate least-squares estimator,
which generally differs from minimum-variance pixel-space weighting.

The script independently computes Snell angles and two scalar ray intersections
to check the production matrix chain. It also checks the exact affine reduction
when radial distortion is zero. These are numerical implementation checks, not
independent experimental validation. Standardized noise draws are reused between
noise levels within a geometry; the resulting CSV rows are not independent.

For an image-to-calibration demonstration run:

```bash
python calibrate.py demo --output work/synthetic_demo
```

This renders 8 synthetic 5 x 6 circular grids with 0.35 mm pitch, extracts
centers, fits 6 training frames, and evaluates 2 held-out frames using disjoint
pose-fitting and evaluation point subsets. Synthetic anchor locations establish
point identities; there is no claim of automatic physical identification.
