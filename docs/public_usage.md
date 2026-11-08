# Usage guide

## Scope

The program has two input paths:

- Quick extraction supports a complete, regular rectangular array of dark dots in 8-bit grayscale images, either from a single camera or from a vertically split bi-prism frame.
- Configuration-driven `detect --config` supports a full-frame ROI, an explicit crop origin, and 16-bit images. It produces point IDs for review; correspondence checking remains necessary.

The two paths use different detection settings. `dot-profile` and
`full_grid_extraction` control the dedicated grid detector in quick extraction;
the configuration-driven path reads `detection`. Changing
`detection.min_area_px` does not change the threshold used by quick `extract`.
The `small` profile is an experimental preset for dots about 7–8 px across;
`legacy` targets larger dots. Neither automatically adapts to arbitrary dot
sizes. Quick extraction also uses fixed candidate-shape and spacing thresholds.
Very small or large dots and strong grid rotations need separate validation.

## Configure new hardware

```bash
python calibrate.py init --template cropped --output ./hardware.json
```

Check at least the following fields. Do not mix mm, px, and degrees.

| Field | Meaning |
|---|---|
| `board.rows`, `board.cols` | Number of dot rows and columns, not squares; the point ID is `row*cols+col`. |
| `board.pitch_mm` | Physical spacing between adjacent dot centers; dot diameter is not required. |
| `prism.refractive_index` | Refractive index at the illumination wavelength, or a justified effective index. |
| `prism.wedge_L_deg`, `wedge_R_deg` | Wedge angle of each physical side, not the sum of the two angles. |
| `prism.thickness_mm` | Maximum thickness in the model, supplied as a fixed input rather than fitted. |
| `camera.pixel_size_um` | A positive scalar such as `3.69` means equal horizontal and vertical pixel size. A pair such as `[3.69, 3.69]` specifies them separately in µm. Result snapshots always store two values. |
| `camera.m_px_per_mm` | Initial signed scale in pixels per millimeter; do not enter the lens's nominal optical magnification directly. |
| `camera.c_px`, `c_source` | Principal point in full-sensor pixels and its source. Quick extraction may use the geometric center as an assumption. |
| `camera.omega_p_deg` | Components of the prism rotation vector in degrees, not XYZ Euler angles. |
| `camera.q_scale_mm` | May be omitted or `null`; it is then derived from the training dot centers and initial intrinsics and held fixed. A manually specified positive value is retained. See `optional_q_scale.md`. |
| `camera.radius_limit_mm` | Radius in reference lens coordinates over which distortion invertibility is checked. |
| `views.left.branch`, `views.right.branch` | Mapping from image-left/image-right views to physical L/R prism branches. |

`m_px_per_mm = signed optical magnification / (horizontal pixel size in µm / 1000)`.
The initial scale can also be estimated from dot spacing at a known pose, but
board tilt affects spacing in the image. Do not adjust refractive index, wedge
angles, or board pitch merely to reduce the residual.

Relative file paths in a configuration are resolved from the configuration
file's directory. Command-line paths are resolved from the current working
directory. Results contain absolute source paths for traceability; redact them
before sharing results publicly.

When quick extraction uses `--config hardware.json`, the same grid must still
be specified with `--rows`, `--cols`, and `--pitch-mm`. A mismatch is an error.
The `--images`, `--pattern`, and suffix options specify the actual image inputs.
For an arbitrary ROI or mask, or custom board coordinates, use the
configuration-driven `detect` path.

## File pairing and coordinates

For example, with `pose01_left.tif` and `pose01_right.tif`:

```bash
python calibrate.py extract --mode prism --images ./images --output ./work/centers \
  --rows 9 --cols 6 --pitch-mm 3 --pattern '*.tif' \
  --left-suffix _left --right-suffix _right --require-complete-pair
```

For an unscaled vertical midpoint split, right-view coordinates are restored as
`u_full = u_crop + left-view width`. Both views must have the same height; the
right view may have the same width as the left or be one pixel wider. Missing or
duplicate pairs, mismatched suffixes, and changes in dimensions are errors. If
the grid is incomplete, `--require-complete-pair` excludes the entire pair. It
does not fill missing points or delete the original files. The presence of a
review image does not mean that its frame was retained.

## Review and fit

```bash
python calibrate.py review --source ./work/centers --output ./work/review --full-resolution
python calibrate.py fit --source ./work/centers --output ./work/fit \
  --confirm-order --show-jkl --backend sparse --workers 4 --starts 5
python calibrate.py audit-points --calibration ./work/fit --output ./work/audit --top 20
```

Before confirming the order, check that equal point IDs in the two views refer
to the same physical dot. Automatic sorting from the image's top row and left
column does not establish a unique physical numbering. Pay particular
attention to large rotations and flips of a symmetric board.

`--workers` parallelizes grouped numerical differentiation; it does not run
several complete calibrations at once. Starting points are processed in
sequence. More threads are not always faster; try four first. `--starts 1` is
useful for a quick check but does not replace a stability analysis. BLAS uses
one thread by default, unless the user has set the corresponding environment
variables. Avoid excessive nested threading. The `auto` backend uses the sparse
path when there are more than 50 training frames or `workers > 1`.

Quick single-camera calibration retains its own data format and three radial
distortion terms. The bi-prism model uses two shared radial terms. Older
extraction directories remain readable; after moving to another computer,
check the original image paths in the configuration.

## Limits of blur filtering

For large dots, `filter-blur` measures the 10%–90% edge-transition width along
image normals. It does not directly measure depth of field or center accuracy.
For small dots, the older ±10 px normal can cross the opposite edge, so the
`small` profile rejects automatic filtering. Unmeasurable large-dot points are
retained for review, and the command returns a non-success status.
`point_count_ready` only means that enough points remain;
`quality_review_required` means that manual review is still needed. Do not
label unmeasured points as sharp, or claim an accuracy gain by comparing RMS
values computed on different point sets before and after filtering.

## Reproducibility and privacy

Fix the training and validation sets and validation points. Record the
configuration snapshot, dependency versions, random seed, and fit settings.
Held-out calibration-board data do not provide independent truth for the
mechanical angles. Measured accuracy still requires an independent reference.

The source exporter reads only allowlisted files and writes a new ZIP; it does
not modify experimental data:

```bash
python tools/export_public.py --output ./dist/public-source.zip
```

The exporter does not overwrite an existing ZIP; use another filename for a
new export. The source is maintained in a separate Git repository. Before a
commit, inspect `git status` and the output of
`python tools/audit_git_tree.py`. Confirm the source rights, attribution, and
release checklist. The license is in the repository-root `LICENSE`. Do not add
local experiment directories to a commit. The allowlist and basic scan are
not a complete secret audit or a substitute for source-provenance and
copyright review.

Exit codes: `0` means the command completed; `2` indicates a configuration or
input error; `3` means that fitting did not pass or filtering needs review;
`130` means directory selection was canceled. In a headless environment,
specify all path arguments explicitly.
