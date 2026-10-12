# unpile

unpile corrects susceptibility distortion in echo-planar images: diffusion b=0 volumes, single-band
reference images, BOLD means. It is one Rust binary. It writes the correction as ITK transforms, so
qsiprep and any ANTs-based pipeline can apply it.

It is built for two situations:

- **The fieldmap is imperfect.** A dual-echo gradient-echo fieldmap from the same session rarely
  lines up with the EPI exactly. The subject moved between the two scans, the shim was not quite
  the same, and phase unwrapping leaves small errors. unpile keeps the fieldmap as a prior and
  refines its position, a low-order shim term and a smooth residual against the EPI itself. See
  {doc}`user/gre-refinement`.
- **There is no fieldmap.** unpile estimates the field from the EPI and an undistorted reference: a
  synthetic b=0 from SynB0, or the subject's T2w with a rigid pose fitted at the same time. See
  {doc}`user/fieldmapless`.

Pairs of opposite phase-encoding direction are also supported ({doc}`user/opposite-polarity`).

The [susceptibility chapter](https://pennlinc.github.io/diffusion-book/book/preprocessing/susceptibility-distortion) of *Diffusion MRI, executed* explains the artifact itself. These
pages cover what unpile does and what to expect. All numbers come from runs you can repeat: simulated
data with ground truth from [TRXScan](https://github.com/PennLINC/TRXScan) and a small set of
OpenNeuro data ({doc}`user/validation`).

```{figure} _static/gre_refinement.png
:alt: A simulated fieldmap acquired after a head movement, the refined field, the true field and the difference.

A simulated fieldmap acquired after a 5.4° / 4 mm head movement (left), the same fieldmap after
unpile refined it against the EPI (second), the true field (third) and the difference (right).
The movement is recovered to 0.8° / 0.6 mm and the field to 0.998 correlation.
```

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

user/gre-refinement
user/fieldmapless
user/opposite-polarity
user/inputs-outputs
user/performance
user/deployment
user/validation
```

```{toctree}
:maxdepth: 1
:caption: Reference

cli/index
development
changes
license
```
