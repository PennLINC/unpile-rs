# unpile

unpile corrects susceptibility distortion in echo-planar images (EPI): diffusion b=0 volumes,
single-band references, BOLD means. It is a single Rust binary that writes its correction as ITK
transforms, so it slots into qsiprep and any ANTs-based pipeline. It is built for two situations:

- **A fieldmap was measured, but not where the EPI is.** A dual-echo gradient-echo (GRE) fieldmap
  acquired after the subject moved, in another session, or under a different shim is brought onto
  the EPI by estimating its rigid pose, a shim polynomial and a smooth residual against the EPI
  itself, rather than applied as acquired. See {doc}`user/gre-refinement`.
- **No fieldmap at all.** The field is estimated from the EPI and an anatomical reference: a
  synthetic undistorted b=0 (SynB0), or the subject's T2w with a joint rigid pose, through a
  physical forward model of the distortion. See {doc}`user/fieldmapless`.

Pairs of opposite phase-encoding polarity are also supported ({doc}`user/opposite-polarity`), so one
tool covers whatever a dataset provides.

The physics of the artifact, why pile-up and stretching happen, and how the three families of
correction relate are explained in the companion book, [Diffusion MRI, executed](https://pennlinc.github.io/diffusion-book/book/preprocessing/susceptibility-distortion); these
pages say what unpile does with that physics and what to expect from it. Every number quoted here
comes from a run that is reproducible from the repository: simulated datasets with ground truth
from [TRXScan](https://github.com/PennLINC/TRXScan), and a small bundle of OpenNeuro data
({doc}`user/validation`).

```{figure} _static/gre_refinement.png
:alt: GRE refinement on a simulated fixture: the fieldmap as acquired after a head movement, the refined field, the true field and their difference.

A measured fieldmap acquired after a 5.4° / 4 mm head movement (left), refined by unpile against the
EPI (second), the simulator's true field (third) and the residual (right). The pose is recovered to
0.8° / 0.6 mm and the field to 0.998 correlation.
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
