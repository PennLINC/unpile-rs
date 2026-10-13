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

## How refinement works, on one line

This section follows a refinement on the same line as {doc}`how-it-works`, so that each term can be
seen doing its job. The fieldmap is made from the true field with four errors that real fieldmaps
have:

- Low resolution: the field is averaged in 9 mm samples along the line, as a fieldmap acquired
  at a coarser resolution than the EPI;
- Misplacement: the fieldmap is moved 4 mm anterior, as after head movement between the two
  scans;
- A different shim: 12 Hz are added everywhere, plus a gradient of 0.32 Hz/mm along the line;
- Noise: 3 Hz of random error inside the head; outside the head the fieldmap is masked.

The reference is the clean b=0, so the comparison is between images of the same contrast. Positive
displacements point anteriorly; on this line 1 Hz moves signal 0.17 mm posteriorly.

```{figure} ../_static/fieldmap_line_input.svg
:alt: The degraded fieldmap against the true field along the line, and the acquired line against the line predicted with the fieldmap as given.

(A) The fieldmap as given and the true field along the line. (B) The acquired b=0, and the
reference pushed forward with the fieldmap as given.
```

The shim error tilts the fieldmap: it is too low at the back of the head and too high at the front.
In (B) the predicted pile-up is too bright, and the frontal edge of the brain is predicted several
voxels too far posterior. This comparison is what refinement works from: it changes the fieldmap
until the predicted line matches the acquired one.

Each term can correct one kind of error. The pose slides the fieldmap along the line, like a
transparency moved until it lines up with the image below; on one line it is a single shift, while
in the volume it is a full rigid transformation. The shim adds a constant, a linear and a
quadratic function of position. These are the fields shim coils produce, so a different shim is
corrected exactly by them. The residual is a smooth correction, kept small by its ridge penalty,
for whatever the other two cannot express.

```{figure} ../_static/fieldmap_line_refine.svg
:alt: The field used and the true displacement, and the prediction against the acquired line, for the fieldmap as given, after pose and shim, and after pose, shim and residual.

The fieldmap as given (top), after refining pose and shim (middle), and after refining all three
terms (bottom). Left: the displacement from the field used, against the truth. Right: the line
predicted with that field, against the acquired b=0.
```

Refining pose and shim reduces the RMS displacement error inside the brain from 3.08 mm to 0.53 mm.
The fieldmap is moved back 4.8 mm, against the 4 mm applied. The shim terms found are −14 Hz and
−0.31 Hz/mm, against the −12 Hz and −0.32 Hz/mm needed.

```{figure} ../_static/fieldmap_line_terms.svg
:alt: The change each term made to the fieldmap, against the change needed to reach the true field.

What each term changed when all three are refined, against the change needed to turn the fieldmap
into the true field (all in Hz).
```

Adding the residual makes this line slightly worse (0.89 mm), and the shift grows to 6.6 mm. Along
one line, a shifted field and a smooth correction can look alike, so the residual and the pose
trade off against each other. In the volume, the pose must fit every line at once, which separates
them. This is why the default refines pose and shim only, and why the residual is added only with a
same-contrast reference (see below).

Taking the errors one at a time shows which terms help and which do not. The table gives the RMS
displacement error inside the brain, with the shift found in parentheses.

| error in the fieldmap | as given | pose + shim | pose + shim + residual |
|---|---|---|---|
| 9 mm resolution | 0.21 mm | 0.27 mm | 0.41 mm |
| moved 4 mm anterior | 1.04 mm | 0.29 mm (−4.3 mm) | 0.39 mm (−4.9 mm) |
| different shim | 3.14 mm | 0.37 mm | 0.40 mm |
| noise, 3 Hz inside the head | 0.50 mm | 0.60 mm | 0.43 mm |
| noise, 15 Hz outside the head, unmasked | 0.00 mm | 0.59 mm | 0.56 mm |
| all four | 3.08 mm | 0.53 mm | 0.89 mm |

- Misplacement and a different shim are what refinement is for: both are corrected to about 0.3 mm.
- A smooth fieldmap that is already in place gains nothing. Refinement can only add error, here
  0.06 mm with pose and shim.
- Voxel-to-voxel noise is not removed. It passes into the field, and the terms adjust to it
  slightly. Smooth a noisy fieldmap before using it.
- Mask the fieldmap outside the head. On this line, noise outside the head alone, with a perfect
  field inside, led refinement 0.6 mm astray. Noisy values there jitter the predicted position of
  the bright scalp edge, and the pose and shim bend to follow it.

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
