# Changes

## 0.2.0 (unreleased)

- Fieldmap refinement without `delta` (the default `--fieldmap-refine pose,shim`) now moves the
  fieldmap's pose and shim. The unrefined residual was frozen with a very large penalty, which made
  the problem so stiff that the line search barely moved the other terms; it is now held at zero by
  dropping its gradient.
- With a fieldmap and `--pose`, only the change to the fieldmap's in-brain mean is pinned, not the
  mean of the whole field, which the shim constant used to cancel.
- With a T2w reference, refining the fieldmap's `pose` alone is recommended unless the shim changed:
  under the cross-contrast metric the shim terms drift (simulated field correlation 0.997 with pose
  alone, 0.960 with pose and shim).
- `--truth-fieldmap-xfm` scores the fieldmap's pose; `--truth-xfm` scores only the reference's pose.
  Both were scored against `--truth-xfm` before.
- `unpile.prepare(reference_xfm=...)` applies an initial ITK transform (`.txt` or ANTs `.mat`) to
  the reference and returns a copy; `--reference-init-xfm` (`reference_init_xfm=`) then makes the fit
  write `pose_composed_xfm.txt`, from the EPI grid to the original reference, and score
  `--truth-xfm` against it. With `--pose` the fit also writes `reference_posed.nii.gz`.
  `unpile.read_itk_affine` and `unpile.write_itk_affine` read and write ITK affines as RAS maps; the
  conventions are tested against nitransforms and were checked against ANTs.
- Documentation: a single-line walkthrough of the estimation (`how-it-works`) and of fieldmap
  refinement, with the PyTorch line model and figure scripts in `docs/tools`.

## 0.1.0 (2026-10-12)

- First release: GRE fieldmap refinement (pose, shim, residual), fieldmap-less correction against
  SynB0 or a T2w with a joint pose, opposite-polarity pairs; ITK warp and pose outputs; CPU backend
  with SIMD and rayon, wgpu GPU backend; truth-scored fixture tests on TRXScan recipes.
