# Fieldmap-less correction

Without a fieldmap, the field has to come from the EPI and an undistorted reference. The book calls
this the [fieldmap-less approach](https://pennlinc.github.io/diffusion-book/book/preprocessing/susceptibility-distortion#correction-step-by-step): warp the EPI along the phase-encoding
axis until it matches an anatomical image. unpile does this with a physical model of the readout
rather than a general deformable registration. The estimated field moves and stretches the
undistorted image the way the readout does, including the pile-up where tissue is compressed, and
the fit compares that prediction with the EPI.

## References

same contrast, `--metric ssd`
: a synthetic undistorted b=0 from SynB0, which qsiprep generates from the T1w. The sum of squared
  differences with a fitted intensity gain. This is the well-posed case. On the simulated dataset
  the field reaches 0.994 correlation with the truth.

T2w, `--metric lncc --pose`
: the subject's T2w on the EPI grid. Local normalized cross-correlation in 5-voxel windows compares
  the two contrasts, and a rigid position of the T2w is fitted together with the field. The position
  is written out as an ITK affine. A raw T2w works when the EPI has T2-like contrast: an SBRef, a
  BOLD mean, a b=0 at a long echo time. It recovers the shape of the field well and about half of
  its amplitude where the field is strongest, so treat it as a refiner of a good starting point.

T1w
: not recommended as the direct reference. Its contrast is close to the inverse of the EPI's and
  the fit fails on real data. Use it through SynB0.

## Fit on a finer grid

The anatomical image has smaller voxels than the EPI. Resampling it down to the EPI grid throws
away the detail the cross-contrast metric needs. Fitting on a grid at about two thirds of the EPI
voxel keeps it:

```{figure} ../_static/resolution_sweep.png
:alt: Field correlation against the fitting grid for a simulated dataset and a real SBRef.

The T2w fit scored on the native grid in the same mask. Two thirds of the EPI voxel (dark bars)
gives the gain on both datasets at 2.4 to 3.4 times the voxels. Finer grids cost 4 to 16 times the
voxels for nothing more. A same-contrast fit does not change at all.
```

Resample the EPI, the mask and the reference onto that grid and pass the acquired phase-encoding
voxel size with `--acq-voxel-mm`. Hz times readout time is a shift in acquired voxels, and the shift
per grid voxel scales with the ratio of voxel sizes.

```bash
unpile fit --epi sbref_1p8mm.nii.gz --reference t2w_1p8mm.nii.gz --mask brain_1p8mm.nii.gz \
    --pe-dir j- --total-readout-time 0.0525 --acq-voxel-mm 2.4 \
    --metric lncc --pose --lam-mm 60 --out out
```

## What to expect

| data | reference | field correlation | rms error |
|---|---|---|---|
| simulated 3 mm, vs truth | clean b=0 | 0.994 | 0.47 mm |
| simulated 3 mm, vs truth | T2w, native grid | 0.54 | 3.9 mm |
| simulated 3 mm, vs truth | T2w, 2 mm grid | 0.76 | 2.7 mm |
| 1.7 mm diffusion b=0, 92 ms readout (HBCD protocol), vs the session's GRE fieldmap | T2w, native grid | 0.87 | 2.4 mm |
| the same, vs topup from the AP/PA pair | T2w, native grid | 0.88 | 2.1 mm |
| 2.4 mm SBRef (ds002737), vs topup from a spin-echo pair | T2w, 1.8 mm grid | 0.74 | 1.6 mm |
| 2 mm spin-echo EPI (BOLD5000), vs topup from a spin-echo pair | T2w, native grid | 0.85 | 1.6 mm |

On the simulated dataset the fitted position recovers a 7.1° / 5.1 mm misplacement of the T2w to
1.3° / 2.0 mm, or 0.5° / 1.6 mm on the 2 mm grid.

```{figure} ../_static/csdsi2_fieldmapless_before_after.svg
:alt: A 1.7 mm diffusion b=0 with a 92 ms readout before and after fieldmap-less correction, alternating, with the white-matter boundary.

The diffusion b=0 from the table before and after correction with the T2w, alternating every two
seconds, through the slices where the displacement is largest. The red line is the white-matter
boundary from a SynthSeg segmentation of the T2w. The displacement is 2.5 mm at the median voxel of
this slice; the frontal pile-up in the acquired image is the compressed signal the model restores.
```

```{figure} ../_static/fieldmapless_sbref.png
:alt: SBRef before and after correction with the white-matter boundary overlaid, and the field next to a topup field.

The ds002737 run from the table, with the white-matter boundary from a SynthSeg segmentation of the
T2w drawn on both EPI panels. Where the field is moderate the cortex lands on it after correction.
In the frontal and temporal hot spots unpile's field has about half of topup's amplitude, so those
regions are only partly restored.
```

```{figure} ../_static/fieldmapless_before_after.svg
:alt: A 2 mm spin-echo EPI before and after correction, alternating, with the white-matter boundary from the T2w.

The BOLD5000 run from the table as a before/after view through the slices with the largest displacement, alternating every two seconds, with the white-matter boundary from a SynthSeg
segmentation of the T2w. The displacement is 1 mm at the median voxel and 4 mm at the 90th percentile.
```
