# unpile fit

Estimate the field from one EPI image and a reference, from an EPI image and a GRE fieldmap, or from two images of opposite phase-encoding polarity. See {doc}`../user/gre-refinement` and {doc}`../user/fieldmapless`.

```text
Estimate the field: one EPI against a reference (optionally with a GRE prior), or two EPIs of opposite polarity

Usage: unpile fit [OPTIONS] --epi <EPI> --mask <MASK> --out <OUT> --total-readout-time <TOTAL_READOUT_TIME> --pe-dir <PE_DIR>

Options:
      --device <DEVICE>
          compute device: flex (pure-Rust CPU) or wgpu (Vulkan/Metal/WebGPU, needs the `wgpu` feature) [default: flex]
      --epi <EPI>
          the distorted EPI image (a b=0 mean, an SBRef, a BOLD mean); every other image must be on its grid
      --reverse-epi <REVERSE_EPI>
          the same EPI acquired with the opposite phase-encoding direction (--reference is then ignored)
      --threads <THREADS>
          CPU threads for the flex backend (default: all cores; qsiprep passes its per-process allocation)
      --membership <MEMBERSHIP>
          voxel membership kernel of the pushforward: tent (trilinear hat, fast) or gauss (the prototype's) [default: tent]
      --reference <REFERENCE>
          undistorted reference on the EPI grid: same contrast (SynB0, a clean b=0) with --metric ssd, or a T2w with --metric lncc
      --pose
          estimate a rigid pose of the reference jointly (single blip)
      --blip-pose
          two-blip mode: also estimate a rigid pose of the second image
      --blip-gain
          two-blip mode: also estimate an intensity gain of the second image
      --fieldmap <FIELDMAP>
          GRE fieldmap in Hz on the EPI grid, used as a prior and refined against the EPI
      --fieldmap-refine <FIELDMAP_REFINE>
          which fieldmap terms to refine: a rigid pose of the fieldmap, a harmonic shim polynomial, a smooth residual (delta; add it only with a same-contrast reference); "none" applies the fieldmap as given [default: pose,shim]
      --fieldmap-ridge <FIELDMAP_RIDGE>
          ridge weight keeping the residual close to the fieldmap [default: 1]
      --lam-mm <LAM_MM>
          penalty in displacement units: lam = lam_mm * (TRT * vox_pe)^2 (overrides --lam)
      --mask <MASK>
          brain mask on the EPI grid (the data term is evaluated inside a dilation of it)
      --out <OUT>
          output directory: field_hz, disp_pe_mm, corrected, sdc_warp (ITK), pose transforms, score.json
      --total-readout-time <TOTAL_READOUT_TIME>
          TotalReadoutTime of the EPI in seconds, as in the BIDS sidecar
      --acq-voxel-mm <ACQ_VOXEL_MM>
          PE voxel size of the ACQUISITION (mm) when the images were resampled onto another grid: Hz x TRT is a shift in acquired voxels, so the shift per grid voxel scales by acq/grid
      --pe-dir <PE_DIR>
          PhaseEncodingDirection of the EPI, as in the BIDS sidecar: i, i-, j, j-, k or k-
      --levels <LEVELS>
          control-grid spacings in mm, coarse to fine [default: 24,12,6,3]
      --lam <LAM>
          bending-energy penalty in field units (prefer --lam-mm) [default: 0.1]
      --iters <ITERS>
          L-BFGS iterations per level and round [default: 60]
      --metric <METRIC>
          data term: ssd for a same-contrast reference, lncc (local normalized cross-correlation) for a T2w [default: ssd]
      --model <MODEL>
          forward model: push (mass-conserving pushforward) or pull_lag (Lagrangian unwarp-and-compare) [default: push]
      --window <WINDOW>
          LNCC window in voxels [default: 5]
      --truth-field <TRUTH_FIELD>
          true field (Hz, on the EPI grid) to score against: TRXScan truth or a reference method's field
      --truth-xfm <TRUTH_XFM>
          ITK affine (.txt/.mat/.h5) the reference pose is scored against (fixed EPI -> moving reference)
  -h, --help
          Print help
```
