# Optional radial-distortion normalization scale

camera.q_scale_mm may be omitted or set to JSON null. It does not need to be measured, and dot centers do not need to be extracted again. Both fit and python -m bbs_calib calibrate support this option. An existing positive value is retained; zero, negative, string, Boolean, and nonfinite values are rejected.

Automatic calculation occurs after the complete-frame training/validation split and uses only accepted training centers. Centers have already been restored to full-sensor pixel coordinates. The calculation is:

    q_initial = solve(K_initial, pixel_full_frame - c)
    q_scale_mm = max(norm(q_initial))

camera.K_init has priority for K_initial. Otherwise, the existing magnification, tilt initial values, and pixel-size ratio are used. Both positive and negative magnification are supported; no extra L/R branch mapping is inferred. The result is a fixed numerical scale, not a newly estimated hardware dimension or a mechanical-parameter optimization variable. Later stages and multistart trials do not recompute it. Center outliers and incorrect initial principal point or magnification can still affect the scale, so it does not replace center review or reasonable intrinsic initialization.

camera.radius_limit_mm may likewise be omitted or set to null; its default is 2*q_scale_mm. An explicitly supplied positive value is retained. Initialization, validation, and reconstruction use the same inverse-distortion domain, and older results lacking the new state fields remain compatible. The default range is an algorithmic check domain, not evidence that the entire camera field has been experimentally validated. A fitted radial function that is nonmonotonic in this range, or a reconstructed point outside the invertible range, is still rejected rather than relaxing the check.

The automatically chosen value and its source are printed in the terminal. Results are saved as:

- calibration.json → distortion.q_scale_mm and state.q_scale: numerical values.
- calibration.json → distortion.normalization: source, training frames used, initial intrinsics, and radius statistics.
- config_snapshot.json → camera.q_scale_mm: resolved numerical value, fixed when the snapshot is reused.

The distortion formula remains qd=q*(1+kappa1*r2+kappa2*r2*r2), where r2=norm(q)^2/q_scale_mm^2. optimization.radial_bounds still constrains coefficients under this normalization, rather than a scale-independent physical distortion limit. Kappa values obtained with different automatic scales cannot be compared directly. To change the scale from a to b while preserving the same physical distortion, multiply the two coefficients by (b/a)^2 and (b/a)^4, respectively, and transform their bounds in the same way. Do not change only the scale in a fitted result; recalibration or a complete parameter transformation is needed to keep the model consistent.

Other hardware inputs and views.left.branch / views.right.branch must still be configured explicitly. This option does not infer them.
