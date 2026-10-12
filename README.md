# unpile

Susceptibility distortion correction for EPI: diffusion b=0 volumes, single-band reference images,
BOLD means. One Rust binary that writes ITK transforms, so qsiprep and ANTs can apply the result.

Documentation: https://unpile.readthedocs.io

unpile does two jobs:

- **Fieldmap refinement.** A gradient-echo fieldmap from the same session is kept as a prior, and its
  position, a low-order shim term and a smooth residual are fitted against the EPI. A fieldmap
  acquired after the subject moved, or with a slightly different shim, is not applied as is.
- **Fieldmap-less correction.** Without a fieldmap, the field is estimated from the EPI and an
  undistorted reference: a SynB0 image, or the subject's T2w with a rigid position fitted at the
  same time.

Pairs of opposite phase-encoding direction are supported too.

## Install

```bash
pip install unpile
```

or build from source (`rust-toolchain.toml` pins the toolchain):

```bash
cargo build --release --features wgpu
```

## Example

The sidecar values are passed as they are: `PhaseEncodingDirection` and `TotalReadoutTime`.

```bash
unpile fit --epi b0.nii.gz --reference t2w.nii.gz --mask brain.nii.gz \
    --pe-dir j- --total-readout-time 0.0557 --metric lncc --pose --lam-mm 60 --out out
antsApplyTransforms -d 3 -i b0.nii.gz -r b0.nii.gz -t out/sdc_warp.nii.gz -o b0_unwarped.nii.gz
```

The same from Python: `unpile.fit(epi, mask, out, sidecar=..., reference=..., metric="lncc", pose=True)`;
`unpile.prepare` puts the inputs on the EPI grid first.

With a fieldmap in Hz on the EPI grid, add `--fieldmap gre_hz.nii.gz`. With a second image of the
opposite direction, use `--reverse-epi` instead of a reference. `--device wgpu` runs on a GPU;
`--threads N` sizes the CPU thread pool.

Outputs: the field in Hz, the displacement in mm, the corrected image, the ITK warp
(`sdc_warp.nii.gz`) and the fitted positions as ITK affines.

## Development

```bash
cargo test --release --features wgpu                                  # unit tests
./target/release/unpile check --dir testdata/rpe                      # operators vs the PyTorch dumps
python scripts/make_fixtures.py phasediff rpe                         # simulated test data (TRXScan)
UNPILE_FIXTURES=$PWD/testdata/fixtures cargo test --release --features wgpu --test fixtures
scripts/fetch_testdata.sh                                             # OpenNeuro test data bundle
```

See `docs/development.md` for the rest.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
