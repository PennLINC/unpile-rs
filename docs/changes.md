# Changes

## 0.2.0 (2026-10-14)

- Fieldmap refinement without `delta` (the default `--fieldmap-refine pose,shim`) now moves the
  fieldmap's pose and shim. The unrefined residual was frozen with a very large penalty, which made
  the problem so stiff that the line search barely moved the other terms; it is now held at zero by
  dropping its gradient.
- With a fieldmap and `--pose`, only the change to the fieldmap's in-brain mean is pinned, not the
  mean of the whole field, which the shim constant used to cancel.
- With a T2w reference, refining the fieldmap's `pose` alone is recommended unless the shim changed:
  under the cross-contrast metric the shim terms drift (simulated field correlation 0.997 with pose
  alone, 0.960 with pose and shim).
- The fieldmap pose refinement is guarded. A refined pose beyond 15 degrees or 15 mm is not a
  registration residual but the optimizer removing a prior the reference cannot support as a whole:
  a synthetic b=0 carries part of the distortion of the b=0 it was made from, so the data term
  prefers a weaker field than the fieldmap, which a pose can only provide by sliding the fieldmap off
  the grid (on the simulated phasediff fixture `--fieldmap-refine pose` against a SynB0 reference
  reached 106 degrees and 200 mm). The pose is reset and no longer refined, the run warns,
  `score.json` records the pose it reached under `gre_pose_divergence`, and the fieldmap's pose is
  applied as given. Against a SynB0 reference apply the fieldmap as given (`--fieldmap-refine none`).
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
