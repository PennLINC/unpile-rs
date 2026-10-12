# unpile fit

Estimate the field from one EPI image and a reference, from an EPI image and a GRE fieldmap, or from two images of opposite phase-encoding polarity. See {doc}`../user/gre-refinement` and {doc}`../user/fieldmapless`.

```text
Estimate the field: one EPI against a reference (optionally with a GRE prior), or two EPIs of opposite polarity

Usage: unpile fit [OPTIONS] --obs <OBS> --mask <MASK> --out <OUT> --trt <TRT>

Options:
      --device <DEVICE>              compute device: flex (pure-Rust CPU) or wgpu (Vulkan/Metal/WebGPU, needs the `wgpu` feature) [default: flex]
      --obs <OBS>                    the distorted EPI image (a b=0 mean, an SBRef, a BOLD mean); every other image must be on its grid
      --pa <PA>                      second blip with the opposite polarity (two-blip mode; --reference is then ignored)
      --threads <THREADS>            CPU threads for the flex backend (default: all cores; qsiprep passes its per-process allocation)
      --membership <MEMBERSHIP>      voxel membership kernel of the pushforward: tent (trilinear hat, fast) or gauss (the prototype's) [default: tent]
      --reference <REFERENCE>        undistorted reference on the EPI grid: same contrast (SynB0, a clean b=0) with --metric ssd, or a T2w with --metric lncc
      --pose                         estimate a rigid pose of the reference jointly (single blip)
      --blip-pose                    two-blip mode: also estimate a rigid pose of the second image
      --blip-gain                    two-blip mode: also estimate an intensity gain of the second image
      --gre <GRE>                    GRE fieldmap (Hz, on the EPI grid, our sign convention) used as a prior
      --gre-refine <GRE_REFINE>      which GRE terms to refine: a rigid pose of the fieldmap, a harmonic shim polynomial, a smooth residual (delta) [default: pose,shim,delta]
      --gre-ridge <GRE_RIDGE>        ridge weight keeping the residual close to the GRE prior [default: 1]
      --lam-mm <LAM_MM>              penalty in displacement units: lam = lam_mm * (TRT * vox_pe)^2 (overrides --lam)
      --mask <MASK>                  brain mask on the EPI grid (the data term is evaluated inside a dilation of it)
      --out <OUT>                    output directory: field_hz, disp_pe_mm, corrected, sdc_warp (ITK), pose transforms, score.json
      --trt <TRT>                    TotalReadoutTime of the EPI in seconds (from the BIDS sidecar)
      --acq-voxel-mm <ACQ_VOXEL_MM>  PE voxel size of the ACQUISITION (mm) when the images were resampled onto another grid: Hz x TRT is a shift in acquired voxels, so the shift per grid voxel scales by acq/grid
      --pe-axis <PE_AXIS>            voxel axis of phase encoding: 0, 1 or 2 for i, j, k of PhaseEncodingDirection [default: 1]
      --sign <SIGN>                  polarity: -1 for a PhaseEncodingDirection ending in "-" (j-), +1 otherwise; pass as --sign=-1 [default: -1]
      --levels <LEVELS>              control-grid spacings in mm, coarse to fine [default: 24,12,6,3]
      --lam <LAM>                    bending-energy penalty in field units (prefer --lam-mm) [default: 0.1]
      --iters <ITERS>                L-BFGS iterations per level and round [default: 60]
      --metric <METRIC>              data term: ssd for a same-contrast reference, lncc (local normalized cross-correlation) for a T2w [default: ssd]
      --model <MODEL>                forward model: push (mass-conserving pushforward) or pull_lag (Lagrangian unwarp-and-compare) [default: push]
      --window <WINDOW>              LNCC window in voxels [default: 5]
      --truth-field <TRUTH_FIELD>    true field (Hz, on the EPI grid) to score against: TRXScan truth or a reference method's field
      --truth-xfm <TRUTH_XFM>        ITK affine (.txt/.mat/.h5) the reference pose is scored against (fixed EPI -> moving reference)
  -h, --help                         Print help
```
