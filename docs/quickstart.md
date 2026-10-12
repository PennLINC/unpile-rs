# Quickstart

Three runs that take under a minute each on a GPU and a few minutes on four CPU cores. The first two
use a simulated dataset with ground truth; the third applies a correction with ANTs.

## Refine a GRE fieldmap

Generate the `phasediff` fixture (TRXScan simulates a 3 mm diffusion scan, a T1w, a T2w and a dual-echo
GRE fieldmap, with the fieldmap acquired after a recorded head movement) and derive unpile's inputs:

```bash
pip install "trxscan[trx]==0.2.1" nibabel nilearn scikit-image scipy
python scripts/make_fixtures.py phasediff
ls testdata/fixtures/phasediff/unpile
```

`b0_AP.nii.gz` is the mean b=0, `gre_hz.nii.gz` the unwrapped fieldmap in Hz resampled onto the EPI
grid, `clean_b0.nii.gz` the simulator's undistorted b=0, `truth_field.nii.gz` the true field and
`meta.json` the readout time and polarity from the sidecars. Fit with the GRE as a prior:

```bash
D=testdata/fixtures/phasediff/unpile
unpile fit --obs $D/b0_AP.nii.gz --reference $D/clean_b0.nii.gz --mask $D/brain_mask.nii.gz \
    --trt 0.055675 --pe-axis 1 --sign=-1 --lam-mm 5 \
    --gre $D/gre_hz.nii.gz --gre-refine pose,shim,delta \
    --truth-field $D/truth_field.nii.gz \
    --truth-xfm $D/../derivatives/trxscan/sub-pdiff/fmap/sub-pdiff_acq-gre_from-fmap_to-dwi_mode-image_desc-truth_xfm.txt \
    --out out/gre
```

The log ends with the two scores the fixture makes possible, the recovered fieldmap pose against
the recorded movement and the field against the truth:

```text
SCORE gre_pose vs truth: residual rotation 0.77 deg, translation 0.58 mm at the origin (identity would leave 5.42 deg, 4.03 mm)
SCORE field corr 0.998 slope 0.991 rms 0.24 of 4.46 mm
```

`out/gre/` holds the field (`field_hz.nii.gz`), the displacement in mm, the corrected EPI, the ITK
warp and the fieldmap's pose as an ITK affine. {doc}`user/gre-refinement` explains the terms.

## Correct without a fieldmap

Same fixture, now with the T2w as the reference. The T2w was acquired after a different recorded
movement (7.1° / 5.1 mm), so the joint pose has something to find:

```bash
unpile fit --obs $D/b0_AP.nii.gz --reference $D/T2w.nii.gz --mask $D/brain_mask.nii.gz \
    --trt 0.055675 --pe-axis 1 --sign=-1 --metric lncc --window 5 --pose --lam 1 \
    --truth-field $D/truth_field.nii.gz \
    --truth-xfm $D/../derivatives/trxscan/sub-pdiff/anat/sub-pdiff_from-T2w_to-dwi_mode-image_desc-truth_xfm.txt \
    --out out/t2w
```

```text
SCORE pose vs truth: residual rotation 1.30 deg, translation 2.04 mm at the origin (identity would leave 7.14 deg, 5.08 mm)
```

On real data the T2w fit works best on a grid finer than the EPI; {doc}`user/fieldmapless` has the
rule and the numbers, and the figure below is a real 2.4 mm single-band reference from OpenNeuro.

```{figure} _static/fieldmapless_sbref.png
:alt: A single-band reference before and after fieldmap-less correction next to its T2w, and the estimated field next to a topup field from a spin-echo pair.

A 2.4 mm SBRef (OpenNeuro ds002737) corrected against the subject's T2w with a joint pose, fitted on
a 1.8 mm grid. Bottom: unpile's field next to FSL topup's field from the spin-echo pair of the same
session, both with their mean removed. They agree at 0.74 correlation; the temporal lobes are where
they differ.
```

## Apply the correction with ANTs

The warp is a standard ITK displacement field, so any image on the EPI grid is corrected with:

```bash
antsApplyTransforms -d 3 -i epi.nii.gz -r epi.nii.gz -t out/gre/sdc_warp.nii.gz -o epi_unwarped.nii.gz -n BSpline
```

The warp moves the signal back; the intensity rescaling for pile-up and stretching (the Jacobian of
the displacement along the phase-encoding axis) is a separate multiplication, which qsiprep applies
with its own Jacobian module. {doc}`user/inputs-outputs` gives the conventions.
