# Fieldmap refinement

A dual-echo gradient-echo fieldmap measures the field in Hz. Multiplied by the readout time it gives
the displacement, as the [measured-fieldmap approach](https://pennlinc.github.io/diffusion-book/book/preprocessing/susceptibility-distortion#correction-step-by-step) in the book
describes. In practice the fieldmap from a session does not fit the EPI exactly:

- The subject moved between the two scans. The usual fix is a rigid registration of the fieldmap's
  magnitude image to the EPI. The magnitude image has little contrast and the EPI is distorted, so
  that registration is uncertain.
- The shim is not quite the same. A re-shim or a different head position changes the low-order part
  of the field.
- Phase unwrapping leaves small errors near signal voids.

unpile keeps the fieldmap as a prior and refines three things against the EPI:

pose
: a rigid transform of the fieldmap. The EPI's own anatomy decides where the fieldmap belongs.

shim
: a low-order harmonic polynomial added to the field: a constant and eight solid harmonics.

residual
: a smooth field added on top, kept near zero by a ridge penalty.

The fieldmap should come from the same session as the EPI, with the same or a similar shim. unpile
does not make a fieldmap from another session usable; the field itself is different then.

The reference is what the unwarped EPI is compared with: a same-contrast image (a SynB0 image, or
on simulated data the clean b=0) or the subject's T2w with `--metric lncc --pose`.

## Running it

```bash
unpile fit --epi b0.nii.gz --reference synb0.nii.gz --mask brain.nii.gz \
    --pe-dir j- --total-readout-time 0.0557 --lam-mm 5 \
    --fieldmap gre_hz.nii.gz --fieldmap-refine pose,shim,delta --out out
```

The fieldmap must be in Hz on the EPI grid. A positive field moves signal toward the positive
phase-encoding axis for `--pe-dir j` and the other way for `j-`. qsiprep's fieldmap workflow
produces this from a phase-difference or two-phase acquisition. `--fieldmap-refine delta` keeps the
fieldmap in place and only lets it bend; `pose,shim,delta` is the default.

## What to expect

On the simulated `phasediff` dataset, TRXScan records the movement it applied to the fieldmap, so
both the position and the field can be scored:

| quantity | fieldmap as acquired | after refinement |
|---|---|---|
| position error vs the recorded movement | 5.4° / 4.0 mm | 0.8° / 0.6 mm |
| field correlation vs truth | 0.95 | 0.998 |
| field rms error | | 0.24 mm |

```{figure} ../_static/gre_refinement.png
:alt: The fieldmap as acquired, after refinement, the truth, and the difference.

The same run.
```

On real data there is no truth. What can be checked is that the position unpile finds agrees with
a rigid registration of the magnitude image, and that it does, to within a degree, on the data we
have tried. A fieldmap with a large unwrapping failure is beyond the residual term; fix the
unwrapping first.
