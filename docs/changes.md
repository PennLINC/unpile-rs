# Changes

## Unreleased

- Fieldmap refinement without `delta` (the default `--fieldmap-refine pose,shim`) now moves the
  fieldmap's pose and shim. The unrefined residual was frozen with a very large penalty, which made
  the problem so stiff that the line search barely moved the other terms; it is now held at zero by
  dropping its gradient.
- With a fieldmap and `--pose`, only the change to the fieldmap's in-brain mean is pinned, not the
  mean of the whole field, which the shim constant used to cancel.
- `--truth-fieldmap-xfm` scores the fieldmap's pose; `--truth-xfm` scores only the reference's pose.
  Both were scored against `--truth-xfm` before.
- Documentation: a single-line walkthrough of the estimation (`how-it-works`) and of fieldmap
  refinement, with the PyTorch line model and figure scripts in `docs/tools`.

## 0.1.0 (2026-10-12)

- First release: GRE fieldmap refinement (pose, shim, residual), fieldmap-less correction against
  SynB0 or a T2w with a joint pose, opposite-polarity pairs; ITK warp and pose outputs; CPU backend
  with SIMD and rayon, wgpu GPU backend; truth-scored fixture tests on TRXScan recipes.
