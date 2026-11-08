# Changelog

## 0.2.1rc1

- Added an isolated Brown-distortion and transverse-offset implementation for
  the extended calibration model. It has separate calibration and reconstruction
  commands, dependencies, and validation metadata.
- Added a shared affine stereo model as a separate comparison, with its own
  calibration schema and tests.
- Added a seeded coordinate-noise experiment for reconstructed displacement
  and a synthetic image-to-calibration demonstration. Numerical outputs and
  assumptions are recorded under `results/` and `docs/REPRODUCIBILITY.md`.
- Added regression tests and Windows/Linux CI checks for the core, affine, and
  Brown implementations.
- Made JSON reads in two tests explicitly UTF-8 for Windows compatibility.

The core optical equations and optimization behavior are unchanged from the
selected 0.2.0 implementation. The Brown model remains separate from the root
package because its calibration files and model parameters are not compatible
with the core model.
