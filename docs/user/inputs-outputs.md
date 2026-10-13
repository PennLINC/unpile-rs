# Inputs and outputs

## Inputs

Every image must be on the grid of `--epi`: same shape, same voxel-to-world matrix. unpile reads
voxel arrays and refuses an image on another grid instead of resampling it. qsiprep passes images
that already share the EPI grid. For a stand-alone run, `nilearn` or `antsApplyTransforms -r epi.nii.gz`
puts a reference or a fieldmap on it.

Two values come straight from the EPI's BIDS sidecar:

| option | sidecar field |
|---|---|
| `--pe-dir` | `PhaseEncodingDirection`: `i`, `i-`, `j`, `j-`, `k` or `k-` |
| `--total-readout-time` | `TotalReadoutTime`, in seconds |

The direction refers to the voxel axes of the image as stored, the same way the sidecar does, so
no conversion is needed. If the images were resampled onto a different grid, add `--acq-voxel-mm`
with the acquired phase-encoding voxel size ({doc}`fieldmapless`).

## Checking the inputs

`unpile.preflight(epi, mask, reference=..., fieldmap=..., reverse_epi=...)` runs before every fit
from Python and warns about the mistakes the binary cannot see: an input that is not on the EPI
grid; a reference or fieldmap with signal in less than 95 % of the brain mask, which is what a
field of view that stops short of the brain looks like; a reference or reverse-polarity image whose
centre of mass sits more than 10 mm from the EPI's, which is what an unregistered or wrong file
looks like; a fieldmap whose rms is not in the range a field in Hz should have; an implausible mask
volume; NaNs. `unpile.prepare` adds the check that applies before resampling: how much of the
EPI's brain each input's field of view contains, warning below 95 % and refusing below 50 %.

qsiprep and the tools it combines unpile with work on LAS+ images. unpile itself follows the
image's own affine, so it runs on any orientation, but `--pe-dir` refers to the voxel axes of the
file it is given, and a sidecar describes the file as it was converted. Three checks cover this:

- The binary prints a note when the EPI is not LAS+, and `unpile.warn_if_not_las` warns from Python.
- `unpile.prepare` refuses an EPI that is not LAS+ and names the fix. Nothing reorients on its own,
  because a reorientation changes which voxel axis is phase-encode: `unpile.reorient(epi, "LAS",
  pe_dir=...)` does it when you call it, writes a new file, and returns the direction carried through
  the change, so `j-` on the converted file becomes what it is on the reoriented one.
  `unpile.reorient_pe_dir(pe_dir, original, reoriented)` is that translation on its own.
- `unpile.check_polarity(epi, mask, reference, fieldmap, pe_dir, trt)` applies the fieldmap as
  given under the sidecar's polarity and under the opposite one and compares how well each explains
  the EPI. On the simulated dataset the right polarity wins by a margin of 0.82; the wrong one
  doubles the distortion. A reference alone cannot decide the sign: with a free field the opposite
  polarity fits the negated field and reaches the same data term, which is why this check needs a
  fieldmap.

A fieldmap for `--fieldmap` is in Hz on the EPI grid. A positive field moves signal toward the
positive phase-encoding axis for `j` and away from it for `j-`. This is the convention of FSL topup's
`--fout` and of the field TRXScan writes.

## Outputs

`field_hz.nii.gz`
: the field in Hz on the EPI grid. With `--pose` the mean inside the brain is set to zero, because a
  constant field and a shift of the reference along the phase-encoding axis are the same thing. The
  log prints the constant implied by the pose.

`disp_pe_mm.nii.gz`
: the displacement along the phase-encoding axis in mm.

`corrected.nii.gz`
: the EPI with the distortion inverted, including the intensity where tissue was compressed.

`sdc_warp.nii.gz`
: the correction as an ITK displacement field: a 5-D NIfTI vector image, intent 1007, vectors in
  LPS mm. It follows the convention `antsApplyTransforms` uses: a point of the corrected grid is read
  from the distorted image at its displaced position. The Jacobian is not in the warp; apply it as a
  separate multiplication, as qsiprep does.

`pose_xfm.txt`, `gre_pose_xfm.txt`
: the position of the reference and of the fieldmap as Insight Transform File affines, from the EPI
  grid to the reference or the fieldmap. They are written through
  [itk-transforms-rs](https://github.com/PennLINC/itk-transforms-rs), which handles the RAS-to-LPS
  conversion.

`score.json`
: the settings of the run and, with `--truth-field`, `--truth-xfm` (the reference's pose) and
  `--truth-fieldmap-xfm` (the fieldmap's pose), the scores against a known field and transforms.

## Applying the correction

```bash
antsApplyTransforms -d 3 -i epi.nii.gz -r epi.nii.gz -t out/sdc_warp.nii.gz -o epi_unwarped.nii.gz -n BSpline
```

This was checked against ANTs on an axis-aligned grid and on a 23° oblique one. `antsApplyTransforms`
with the warp reproduces unpile's own resampling to 1e-6 inside the brain, and qsiprep's Jacobian of
the warp times that resampling reproduces unpile's intensity-corrected image to the same precision.
`unpile export` writes the same files from an existing field, together with the internally
resampled images, for this comparison.
