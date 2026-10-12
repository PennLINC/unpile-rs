# unpile (Rust)

Susceptibility distortion correction for EPI, built for two jobs: refining a measured GRE fieldmap
against the EPI it is meant to correct, and fieldmap-less correction against an anatomical reference.
It is a Rust port of a PyTorch prototype on [Burn](https://burn.dev),
and it writes its results as ITK transforms that `antsApplyTransforms` and qsiprep consume directly.

The estimator fits an off-resonance field (Hz) on a cubic B-spline control grid, coarse to fine, by
L-BFGS with a strong-Wolfe line search, through a mass-conserving pushforward (or a Lagrangian
pull) forward model:

- **GRE refinement** (`--gre`): a measured fieldmap enters as a prior with its own rigid pose, a
  harmonic shim polynomial and a ridged residual, so a fieldmap acquired after the subject moved,
  or under a different shim, is brought onto the EPI instead of being applied as is.
- **Fieldmap-less** (`--reference`, `--pose`): one EPI against a reference of the same contrast
  (SynB0, a clean b=0) with SSD, or of another contrast (T2w) with local normalized cross-correlation
  and a joint rigid pose of the reference.
- Two images of opposite phase-encoding polarity are also supported, through a latent undistorted
  image solved by conjugate gradients between field rounds.

Autodiff provides every derivative; no hand-written backward kernels are needed, including the
adjoint of the forward operator used by the CG solves.

## CPU performance

The CPU backend (`--device flex`) is built with burn-flex's SIMD and rayon features and sizes its
thread pool with `--threads N`. The pushforward's voxel membership kernel is the tent (trilinear hat)
by default; `--membership gauss` selects the prototype's Gaussian, which costs 2-3x more on the CPU
for the same accuracy. `UNPILE_PROFILE=1` prints inclusive wall time per operation (meaningful on the
CPU backend; the wgpu backend executes lazily). On four cores a 3 mm single-blip fit takes about 25 s
and a 2.4 mm real-data fit at the full schedule about six minutes; see `docs/ci_fixtures.md`.

## Fitting grid

When the anatomical reference is finer than the EPI, fit on a grid at about two thirds of the EPI
voxel (3 mm -> 2 mm, 2.4 mm -> 1.8 mm): resample the EPI, mask and reference onto that grid and pass
`--acq-voxel-mm <EPI voxel>` so Hz x TRT is still read as a shift in acquired voxels. On the pd50
fixture the T2w fit improves from 0.54 to 0.76 correlation with the truth field and the recovered
pose from 1.2 degree / 1.9 mm to 0.7 degree / 1.7 mm; on a real 2.4 mm SBRef from 0.66 to 0.74 against
the SE-pair field. Finer grids cost 4-16x the voxels for no further gain; the clean-reference fit
does not change at all, so the gain is the reference's preserved detail inside the LNCC windows.

## Install

`pip install unpile` (wheels with the wgpu backend; a CPU-only machine uses `--device flex`, the
default) or build from source as below. Test data: `scripts/make_fixtures.py` generates the TRXScan
fixtures and `scripts/fetch_testdata.sh` downloads the OpenNeuro bundle; see `docs/ci_fixtures.md`.

## Build

Burn 0.22 needs rustc 1.95 (`rust-toolchain.toml` pins it for this project only).

    cargo build --release                  # pure-Rust CPU backend
    cargo build --release --features wgpu  # + Vulkan/Metal/WebGPU backend (`--device wgpu`)

## Verify against the prototype

`testdata/rpe/` holds the TRXScan rpe fixture inputs and PyTorch intermediates (splat, pullback,
Gaussian, LNCC/SSD/bending values, posed reference, latent image, CG unwarp):

    unpile --device wgpu check --dir testdata/rpe

All operators agree with PyTorch to float precision (max |diff| <= 4e-5).

## Fit

    unpile --device wgpu fit --obs b0.nii.gz --reference clean.nii.gz --mask brain.nii.gz \
        --trt 0.0557 --pe-axis 1 --sign=-1 --levels 24,12,6,3 --lam 0.1 --out out/run
    unpile --device wgpu fit --obs ap.nii.gz --pa pa.nii.gz --blip-pose --blip-gain --lam-mm 5 ...
    unpile --device wgpu fit --obs b0.nii.gz --reference t2w.nii.gz --metric lncc --pose --lam 1 ...
    unpile --device wgpu fit ... --gre gre_hz.nii.gz --gre-refine pose,shim,delta

Outputs: `field_hz.nii.gz`, `disp_pe_mm.nii.gz`, `corrected.nii.gz` (`corrected_pa.nii.gz` for two
blips), `score.json` (with the truth score when `--truth-field` is given). `--sign` is -1 for a
`j-` acquisition and +1 for `j`; a positive field displaces signal by `sign * field * TRT` voxels
along the PE axis. `--lam-mm` expresses the bending penalty in displacement units
(lam = lam_mm * (TRT * vox_pe)^2), invariant to readout time and resolution.

## ITK output (what qsiprep consumes)

Every `fit` writes the correction as ITK transforms next to the images, in the pull convention
`antsApplyTransforms` uses (a point of the FIXED image is mapped to where its value is read in
the MOVING image):

- `sdc_warp.nii.gz`: the susceptibility displacement as a 5-D NIfTI vector image (intent 1007,
  float32, vectors in LPS mm) on the EPI grid. Fixed = corrected grid, moving = distorted EPI;
  the vector is `sign * field_Hz * TRT` voxel steps along the PE axis, taken to world space by the
  image affine and flipped to LPS, the same convention as qsiprep's `_hz_to_warp` and TOPUP's
  field. The intensity (Jacobian) modulation is not in the warp; qsiprep applies it separately
  (`ModulateByWarpJacobian`), and `jacobian_determinant(sdc_warp, pe_axis)` of that module
  reproduces unpile's own modulated pullback to float precision.
- `pose_xfm.txt` (with `--pose`) and `gre_pose_xfm.txt` (with `--gre`): the estimated rigid
  pose as an Insight Transform File affine, fixed EPI grid -> moving reference / fieldmap. The
  internal pose lives in the PE-last grid's mm coordinates; `xfm::pose_world_affine` conjugates
  it with the voxel-to-world map and `itk-transforms-rs` does the RAS->LPS step
  (`Affine3::into_itk_components`).
- `--truth-xfm <itk .txt|.mat|.h5>` scores a pose against a TRXScan truth transform
  (`*_from-fmap_to-dwi_*desc-truth_xfm.txt`, `*_from-T2w_to-dwi_*`), read through
  `itk_transforms_rs::read_itk`: residual rotation and translation of `estimate o truth^-1`
  (qsiprep's `truth_scoring.rigid_error`), with the identity's residual printed as the baseline.

`unpile export --field field_hz.nii.gz [--obs epi.nii.gz] [--reference ref.nii.gz --pose rx,ry,rz,tx,ty,tz]`
writes the same files from an existing field, plus the internally resampled images
(`pull_nojac.nii.gz`, `pullback.nii.gz` with the Jacobian, `posed.nii.gz`) to compare against
`antsApplyTransforms -t sdc_warp.nii.gz` / `-t pose_xfm.txt`. Verified on the 3 mm LAS fixture
(agreement to 1e-6 relative) and on a 23 degree oblique SBRef grid (1e-6 relative inside the
brain; the only differences are FOV-edge voxels where the two resamplers extrapolate
differently).

The `itk` feature (default) pulls `itk-transforms-rs`, whose HDF5 reader builds a static HDF5;
WebAssembly builds use `--no-default-features --features wgpu` (the warp writer needs nothing
beyond `nifti`, so only the affine output and `--truth-xfm` go away).

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
