# Inputs and outputs

## Inputs

Every image must be on the grid of `--obs`: same shape and same voxel-to-world matrix. unpile
reads voxel arrays and refuses an input on another grid rather than resampling it silently; qsiprep
hands over images that already share the EPI grid, and for a stand-alone run `nilearn` or
`antsApplyTransforms` put a reference or a fieldmap onto it.

The three acquisition parameters come from the EPI's BIDS sidecar:

| option | sidecar | value |
|---|---|---|
| `--trt` | `TotalReadoutTime` | seconds |
| `--pe-axis` | `PhaseEncodingDirection` | 0, 1, 2 for `i`, `j`, `k` |
| `--sign` | `PhaseEncodingDirection` | `-1` when it ends in `-`, else `+1`; written `--sign=-1` |

If the images were resampled onto a grid other than the acquisition's, add `--acq-voxel-mm` with
the acquired phase-encoding voxel size ({doc}`fieldmapless`).

A GRE fieldmap for `--gre` is in Hz on the EPI grid, in unpile's sign convention: a positive field
displaces signal toward the positive phase-encoding axis when `--sign` is `+1`. This is the
convention of the field TRXScan writes and of FSL topup's `--fout`.

## Outputs

`field_hz.nii.gz`
: the estimated field in Hz on the EPI grid. With `--pose` the in-brain mean is pinned to zero,
  because a constant field and a translation of the reference along the phase-encoding axis are
  the same thing; the log reports the constant implied by the pose (`DC from pose`).

`disp_pe_mm.nii.gz`
: the displacement along the phase-encoding axis in mm: `sign × field × TRT × voxel`.

`corrected.nii.gz`
: the EPI restored by inverting the forward model, with the intensity restored where tissue was
  compressed.

`sdc_warp.nii.gz`
: the correction as an ITK displacement field (a 5-D NIfTI vector image, intent 1007, vectors in
  LPS mm), in the pull convention `antsApplyTransforms` uses: a point of the corrected grid is read
  from the distorted image at its displaced position. The Jacobian rescaling is not inside the warp;
  apply it as a separate multiplication, as qsiprep does.

`pose_xfm.txt`, `gre_pose_xfm.txt`
: the reference pose and the fieldmap pose as Insight Transform File affines, from the EPI grid to
  the reference or fieldmap, written through
  [itk-transforms-rs](https://github.com/PennLINC/itk-transforms-rs) so the RAS-to-LPS step is the
  same one the rest of the Rust neuroimaging stack uses.

`score.json`
: the run's settings and, with `--truth-field` and `--truth-xfm`, the scores against a known field and
  transform.

## Applying the correction

```bash
antsApplyTransforms -d 3 -i epi.nii.gz -r epi.nii.gz -t out/sdc_warp.nii.gz -o epi_unwarped.nii.gz -n BSpline
```

Verified against ANTs on an axis-aligned grid and on a 23° oblique one: `antsApplyTransforms` with
the warp reproduces unpile's own resampling to 1e-6 inside the brain, and qsiprep's Jacobian of the
warp times that resampling reproduces unpile's intensity-corrected image to the same precision.
`unpile export` writes the same files from an existing field, together with the internally
resampled images, for exactly this comparison.
