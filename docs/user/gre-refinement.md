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
produces this from a phase-difference or two-phase acquisition.

`--fieldmap-refine` chooses the terms. The default is `pose,shim`. Add `delta` only with a
same-contrast reference (SynB0 or, on simulated data, the clean b=0): under the T2w metric the
residual term pulls a good fieldmap away from the truth. With `delta` on and a same-contrast
reference, the position is recovered too; without it, the position barely moves on real data.

## What to expect

On the simulated `phasediff` dataset TRXScan records the movement it applied to the fieldmap, so
both the position and the field can be scored. With the clean b=0 as the reference and all three
terms:

| quantity | fieldmap as acquired | after refinement |
|---|---|---|
| position error vs the recorded movement | 5.4° / 4.0 mm | 0.8° / 0.6 mm |
| field correlation vs truth | 0.95 | 0.998 |
| field rms error | | 0.24 mm |

```{figure} ../_static/gre_refinement.png
:alt: The fieldmap as acquired, after refinement, the truth, and the difference.

The same run.
```

On real data there is no truth, but there is an independent field: FSL topup's from an AP/PA pair
of the same session. On a 1.7 mm diffusion scan with a 92 ms readout and a same-shim GRE fieldmap
(the HBCD protocol), scored against that field inside the brain:

| fieldmap | reference | terms | correlation / slope / rms vs topup |
|---|---|---|---|
| as acquired (registered magnitude) | | | 0.952 / 0.99 / 1.38 mm |
| as acquired | T2w | pose, shim | 0.958 / 1.02 / 1.35 mm |
| as acquired | T2w | pose, shim, delta | 0.918 / 1.06 / 2.02 mm |
| displaced by 5° / 4 mm | T2w | pose, shim | 0.887 / 0.99 / 2.25 mm (position moved 0.4°) |
| displaced by 5° / 4 mm | topup-corrected b=0 | pose, shim, delta | 0.974 / 1.00 / 1.03 mm (position recovered: 5.1°) |

Two things follow. A fieldmap that was registered properly is already close to the best field
available, and refinement adds little. A fieldmap that is out of place is recovered only with a
same-contrast reference and the residual term; the T2w metric does not see where the fieldmap
belongs. In qsiprep that reference is the SynB0 image. The topup-corrected b=0 stands in for it in
the last row because no SynB0 exists for this subject yet.

```{figure} ../_static/csdsi2_panels.png
:alt: The 1.7 mm b=0 as acquired, corrected without a fieldmap, corrected with the displaced-then-refined fieldmap, and the T2w; below, the four fields.

The last row of the table. Top: the b=0 as acquired, corrected without a fieldmap, corrected with
the displaced fieldmap after refinement, and the T2w, with the white-matter boundary from a SynthSeg
segmentation of the T2w in red. Bottom: the field without a fieldmap, the fieldmap as displaced, the
fieldmap after refinement, and topup's field from the AP/PA pair.
```

```{figure} ../_static/csdsi2_refinement_before_after.svg
:alt: The b=0 before and after correction with the recovered fieldmap, alternating.

The b=0 before and after correction with the recovered fieldmap, alternating every two seconds,
through the slices where the displacement is largest.
```

A fieldmap with a large unwrapping failure is beyond the residual term; fix the unwrapping first.
This fieldmap has one such voxel, a 58 mm spike where topup says 20 mm, and it stays.
