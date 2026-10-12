# Fieldmap-less correction

When no fieldmap was acquired, the field has to come from the EPI and an undistorted reference. The
book describes this as the [fieldmap-less approach](https://pennlinc.github.io/diffusion-book/book/preprocessing/susceptibility-distortion#correction-step-by-step): warp the
EPI along the phase-encoding axis until it matches an anatomical image. unpile does this with a
physical forward model rather than a generic deformable registration: the estimated field
displaces and stretches the undistorted image exactly as the readout does, including the
intensity pile-up where tissue is compressed, and the fit compares that prediction with the
acquired EPI.

## References and metrics

same contrast, `--metric ssd`
: a synthetic undistorted b=0 from SynB0 (which qsiprep generates from the T1w) or, on simulated
  data, the simulator's clean b=0. The sum of squared differences with a closed-form intensity gain.
  This is the well-posed case: on the fixture the field reaches 0.994 correlation with the truth.

T2w, `--metric lncc --pose`
: the subject's T2w, resampled onto the EPI grid. Local normalized cross-correlation in 5-voxel
  windows compares the two contrasts, and a rigid pose of the T2w is estimated jointly with the
  field, because the T2w is never exactly where the EPI is. The pose is written out as an ITK
  affine. A raw T2w carries the field well when the EPI has T2-like contrast (an SBRef, a BOLD mean,
  a b=0 at a long echo time); it is a refiner of a good starting point rather than a replacement
  for a fieldmap, and the numbers below say how much.

T1w
: not recommended as the direct reference: its contrast is close to the inverse of the EPI's and the
  cross-contrast fit fails on real data. Use it through SynB0 instead.

## Fit on a finer grid when the reference is finer

The EPI's voxels are coarse and the anatomical's are fine. Resampling the T2w down to the EPI grid
throws away the detail the cross-contrast metric needs. Fitting instead on a grid at about two
thirds of the EPI voxel, with the EPI, mask and reference resampled onto it, keeps that detail:

```{figure} ../_static/resolution_sweep.png
:alt: Field correlation against the fitting grid for a simulated fixture and a real SBRef.

The T2w fit scored on the native grid in the same mask. Two thirds of the EPI voxel (dark bars)
captures the gain on both datasets at 2.4 to 3.4 times the voxels; finer grids cost 4 to 16 times
the voxels for no further gain. A same-contrast fit does not change at all, which shows the gain
is the reference's preserved detail, not anything in the EPI.
```

Pass the acquisition's phase-encoding voxel size with `--acq-voxel-mm` whenever the images were
resampled: Hz times readout time is a shift in acquired voxels, and the shift per grid voxel scales
by the ratio of voxel sizes.

```bash
unpile fit --obs sbref_1p8mm.nii.gz --reference t2w_1p8mm.nii.gz --mask brain_1p8mm.nii.gz \
    --trt 0.0525 --acq-voxel-mm 2.4 --pe-axis 1 --sign=-1 \
    --metric lncc --window 5 --pose --lam-mm 60 --out out
```

## What to expect

| data | reference | field correlation | rms residual |
|---|---|---|---|
| simulated 3 mm fixture, vs truth | clean b=0 | 0.994 | 0.47 mm |
| simulated 3 mm fixture, vs truth | T2w, native grid | 0.54 | 3.9 mm |
| simulated 3 mm fixture, vs truth | T2w, 2 mm grid | 0.76 | 2.7 mm |
| 2.4 mm SBRef (ds002737), vs the spin-echo pair's topup field | T2w, 1.8 mm grid | 0.74 | 1.6 mm |
| 2 mm spin-echo EPI (BOLD5000), vs the spin-echo pair's topup field | T2w, native grid | 0.85 | 1.6 mm |

The joint pose recovers a recorded 7.1° / 5.1 mm misplacement of the T2w to 1.3° / 2.0 mm on the
fixture (0.5° / 1.6 mm on the 2 mm grid). The honest summary: the T2w path is a refiner, in the
same class as SyN-based fieldmap-less correction, and it is at its best when it starts from a
SynB0 field and sharpens it, or when the EPI itself is T2-weighted.

```{figure} ../_static/fieldmapless_sbref.png
:alt: SBRef before and after correction with the T2w's edges overlaid, and the field next to a topup field.

The ds002737 run from the table, with the T2w's edges drawn on the EPI panels. The correction moves
the cortex onto them where the field is moderate; in the frontal and temporal hot spots unpile's
field has about half of topup's amplitude, so those regions are only partly restored.
```

```{figure} ../_static/fieldmapless_sbref_flicker.gif
:alt: The same slice alternating between the acquired and the corrected SBRef.

The same slice alternating between the acquired and the corrected image, one second each.
```
