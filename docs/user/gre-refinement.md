# GRE fieldmap refinement

A dual-echo gradient-echo fieldmap measures the off-resonance field directly, in Hz, and the
[measured-fieldmap approach]({{ book }}#correction-step-by-step) in the book turns it into a
displacement with the readout time. Two things stand between that measurement and the EPI it is
meant to correct:

- **The head is not in the same place.** The fieldmap is a separate acquisition, often at the end of
  the session or in another session, and the subject moved in between. A rigid registration of the
  fieldmap magnitude to the EPI is the usual fix, but a magnitude image has little contrast and the
  EPI is distorted, so that registration is itself uncertain.
- **The field is not quite the same field.** Re-shimming between scans, or a different head
  position in the magnet, changes the low-order part of the field. Fields measured minutes apart
  on the same subject differ by a smooth polynomial even when the anatomy is identical.

unpile treats the fieldmap as a **prior** and refines three things against the EPI itself:

pose
: a 6-parameter rigid transform of the fieldmap, estimated from how well the field unwarps the EPI
  onto the reference. The EPI's own anatomy judges where the fieldmap belongs.

shim
: a low-order harmonic polynomial (a constant and eight solid harmonics) added to the field, which
  is what a shim change or a head displacement in the magnet does to it.

residual
: a smooth field added on top, kept close to zero by a ridge penalty, for whatever the fieldmap did
  not capture: unwrapping errors near signal voids, a slightly different readout, local differences.

The reference is what the unwarped EPI is compared with: a same-contrast image such as a synthetic
undistorted b=0 from SynB0 or, on simulated data, the clean b=0; or the subject's T2w with the
cross-contrast metric and a joint pose ({doc}`fieldmapless`).

## Running it

```bash
unpile fit --obs b0.nii.gz --reference synb0.nii.gz --mask brain.nii.gz \
    --trt 0.0557 --pe-axis 1 --sign=-1 --lam-mm 5 \
    --gre gre_hz.nii.gz --gre-refine pose,shim,delta --out out
```

The fieldmap must be in Hz, in unpile's sign convention (a positive field moves signal toward the
positive phase-encoding axis for polarity `+1`), and resampled onto the EPI grid. qsiprep's
fieldmap workflow produces exactly this from a phase-difference or two-phase acquisition. Which
terms are refined is chosen with `--gre-refine`: `delta` alone keeps the fieldmap in place and only
lets it bend; `pose,shim,delta` is the default and what the numbers below use.

## What to expect

On the simulated `phasediff` fixture, where TRXScan records the movement it applied to the fieldmap,
the refinement recovers that movement and the field:

| quantity | before (fieldmap as acquired) | after refinement |
|---|---|---|
| fieldmap pose residual vs recorded movement | 5.4° / 4.0 mm | 0.8° / 0.6 mm |
| field correlation vs truth | 0.95 | 0.998 |
| field rms residual | | 0.24 mm |

```{figure} ../_static/gre_refinement.png
:alt: The fieldmap as acquired, after refinement, the truth, and the residual.

The same run as in {doc}`../quickstart`.
```

On real data there is no truth; what can be checked is reproducibility and agreement with
independent estimates. On a dataset with coronal GRE fieldmaps acquired in eight sessions, the
refined fields of different sessions agree to 98%, and the pose unpile finds matches the rigid
registration qsiprep performs on the magnitude image to within a degree, with the advantage that
it is driven by the EPI's anatomy rather than the magnitude image's.

## When it will not help

- A fieldmap with large unwrapping failures: the residual term absorbs small errors, not a region
  that is off by a full wrap. Fix the unwrapping first.
- A fieldmap acquired with a different field of view that does not cover the EPI's brain: the pose
  and shim are estimated where both exist, and the field outside the fieldmap's coverage comes from
  the reference alone.
