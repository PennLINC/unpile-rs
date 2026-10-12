# Development

## Build and test

```bash
cargo build --release --features wgpu          # the binary with the GPU backend
cargo test --release --features wgpu           # unit tests
./target/release/unpile check --dir testdata/rpe   # operators vs the PyTorch dumps
python scripts/make_fixtures.py phasediff rpe  # TRXScan fixtures into testdata/fixtures
UNPILE_FIXTURES=$PWD/testdata/fixtures cargo test --release --features wgpu --test fixtures -- --nocapture
```

`cargo fmt --check` and `cargo clippy --release --features wgpu -- -D warnings` must pass; CI runs
both. The Python package lives in `python/unpile` and is packaged into the wheel by maturin
(`python-source`); `maturin build --release --features wgpu --out dist`, `pip install dist/*.whl`
and `python -m pytest python/tests` exercise it, as the CI wheel job does.

## Documentation

The CLI pages are generated from the binary's `--help` output and the figures from fixture and
test-data fits; regenerate them after changing the CLI or the fits:

```bash
python docs/tools/make_cli_pages.py
python docs/tools/make_figures.py --fixtures testdata/fixtures --testdata-real <ds002737 at 1.8 mm> --real-out <its fit> --wm <white-matter mask> --b5k <BOLD5000 ses-01> --b5k-out <its fit> --b5k-wm <white-matter mask> --csdsi2 <CSDSI2 inputs> --csdsi2-t2w-out <its T2w fit> --csdsi2-gre-out <its fieldmap recovery fit>
sphinx-build -W docs docs/_build/html
```

## Code layout

`src/ops.rs` holds the operators (the pushforward splat, interpolation, box and Gaussian filters,
the LNCC loss, bending energy), `src/field.rs` the B-spline control grid, `src/problem.rs` the data
terms (reference, GRE prior, two-blip latent image), `src/pose.rs` the rigid pose, `src/optimize.rs`
the coarse-to-fine L-BFGS driver with `src/lbfgs.rs`, `src/xfm.rs` the ITK outputs and truth
scoring, `src/io.rs` NIfTI I/O, and `src/profile.rs` the per-operation timer behind
`UNPILE_PROFILE`.

## CI fixtures

Two tiers, as in qsiprep: TRXScan fixtures with ground truth, generated at CI time from a spec and
cached on its checksum, and a small bundle of derived OpenNeuro data hosted as a GitHub release.
The tests lead with the two workflows unpile is for: refining a measured GRE fieldmap (pose, shim,
residual) and fieldmap-less correction against an anatomical reference. Two-blip estimation is
supported and tested, but as a plain mode; the frozen TOPUP fields in the bundle are used as
consistency references, not as a competitor.

### Budget

CI runners have no GPU, so the tests use the pure-Rust CPU backend (`--device flex`, Burn's
`burn-flex`). Three things made it fast enough for per-PR tests (all measured pinned to four cores,
the GitHub Actions runner size; `UNPILE_PROFILE=1` prints the per-operation breakdown):

- Burn's `flex` feature pulls `burn-flex` with only its `std` feature; `Cargo.toml` depends on
  `burn-flex` directly to turn on its `simd` and `rayon` features. Parallelism is per tensor above
  256 K elements, so it engages on the lifted `[offsets, voxels]` splat tensors even at 3 mm.
- burn-flex evaluates `erf`/`exp`/`log` one element at a time through libm (only arithmetic is
  SIMD), and the splat's smoothed ramp was 90% of its cost. The ramp is now evaluated once per cell
  edge and differenced across offsets, and the default membership kernel is the tent (trilinear hat,
  `--membership tent`), whose ramp is a clamped quadratic with no transcendental functions. Accuracy
  is identical to the Gaussian kernel on every fixture, folds included; `--membership gauss` keeps the
  prototype's kernel and `check` always uses it because the stored dumps are Gaussian.
- The LNCC box mean is three separable 1-D filters instead of a 5x5x5 convolution.

`--threads N` sizes the rayon pool (qsiprep passes its per-process allocation). Mesa lavapipe runs
the wgpu backend on the CPU correctly but six times slower than `flex`; Burn's LLVM-JIT `burn-cpu`
backend (`--features cpu`, `--device cpu`) passes the check but its 3 mm fit ran over 30 minutes.

| job (levels 24/12/6 mm, 20 iterations, 4 cores) | before | now | result |
|---|---|---|---|
| operator check against the stored dumps | 17 s | 9 s | 1e-6 |
| 3 mm fixture, single blip, clean reference | 102 s | 25 s | 0.995 / 0.97 / 0.41 mm |
| 3 mm fixture, GRE prior with pose/shim/residual | 83 s | 37 s | 0.997 / 0.99 / 0.32 mm; pose residual 1.7 degree / 0.9 mm from 5.4 / 4.0 |
| 3 mm fixture, T2w LNCC + joint pose (fieldmap-less) | 213 s | 41 s | pose residual 1.3 degree / 2.0 mm from 7.1 / 5.1 |
| 3 mm fixture, two blips | 329 s | 59 s | 0.990 / 0.92 / 0.67 mm |
| ds002737 2.4 mm SBRef, T2w LNCC + pose, short schedule | 530 s | 84 s | not converged at this schedule |
| ds002737 2.4 mm SBRef, T2w LNCC + pose, full schedule (24/12/6/3, 60 it) | 1077 s | 377 s | 0.645 / 0.40 / 1.85 mm |

So the whole per-PR tier (check plus four 3 mm fixtures) is under three minutes of test time, and a
full-schedule real-data fit is about six minutes on four cores, which puts the OpenNeuro bundle
within reach of a per-PR job too; BOLD5000 at 2 mm stays nightly. The 1.7 mm fold fixture remains
a GPU-only local benchmark.

### Tier 1: TRXScan fixtures (truth-scored)

`tests/fixtures/trxscan_fixtures.txt` lists the recipes (TRXScan 0.2.1 parameters only). CI installs
`trxscan==<version>`, fetches the `sub-60501` phantom (TRXScan's own GitHub-release download, cached),
generates each fixture into `$UNPILE_FIXTURES/<name>` and caches the directory on the spec checksum,
as qsiprep does with `.circleci/trxscan_fixtures.txt`.

| fixture | recipe | what is asserted |
|---|---|---|
| `phasediff` | qsiprep's phasediff recipe: GRE acquired after a recorded 5 degree / 4 mm movement | GRE prior with pose + shim + residual refinement: field vs truth; recovered fieldmap pose vs `from-fmap_to-dwi` truth via `--truth-xfm` (reference run: 0.68 degree / 0.85 mm residual from 5.4 / 4.0) |
| `fieldmapless` | AP only, T1w/T2w, no fieldmap | clean-b0 ceiling (SSD) and the T2w LNCC fit with joint pose: field vs truth, pose vs `from-T2w_to-dwi` truth |
| `oblique` | fieldmapless with `acq_rotation` | the same on a rotated header: PE-axis and affine handling, and the exported warp read back (vector = step x field) |
| `rpe` | AP + PA | two-blip fit vs truth and the operator check against the stored torch dumps |
| `fold` | rpe at 3 mm with the echo spacing raised until voxels fold | push vs Lagrangian control inside the J<0.3 and folded regions (the multiplier needs one calibration run; at 1.7 mm x2 folds 2295 voxels) |

The phasediff fixture needs its GRE in Hz on the fieldmap grid; until TRXScan writes that
derivative, the prep step unwraps the simulated phase difference (low noise, few wraps) in Python.
The T2w and the GRE must be resampled onto the EPI grid in the prep step: the CLI refuses inputs on
another grid. Head, texture and realistic GRE noise wait for a TRXScan release from `dwi-head`.

### Tier 2: OpenNeuro bundle (consistency-scored)

`scripts/fetch_testdata.sh` downloads `unpile-testdata-<tag>.tar.gz` from the GitHub release and
checks its sha256; the bundle's `registry.txt` checks every file. Published as release `v1` of
`github.com/PennLINC/unpile-testdata` (42 MB asset, fetch verified 2026-10-11; metadata files tracked in that repo), with `meta.json` (PE axis, polarity, readout time from the
sidecars), `PROVENANCE.md` and `LICENSES.md` (ds001600 CC-BY, ds002737 and BOLD5000 CC0).

| directory | tests |
|---|---|
| `ds001600/` (3 mm GE-EPI, measured dual-echo GRE on the EPI grid, T1w) | GRE-prior refinement on real data: pose and shim stay small, the refined field agrees with the GRE better than the unrefined one; T1w fieldmap-less single blip: agreement with the GRE within a recorded bound |
| `ds002737/` (SBRef pair, SE pair, T1w and T2w, 23 degree oblique) | T1w and T2w fieldmap-less fits on the SBRef vs the SE-pair field (consistency bound); ITK export on a real oblique grid |
| `bold5000/ses-0{1,2}/` (SE and GE pairs, T2w per session) | T2w fieldmap-less fit vs the SE-pair field; cross-session reproducibility of the fieldmap-less field (recorded 0.31 mm); SE-vs-GE consistency (slow tier) |

The `ds001600/ap_resampled_from_3p75mm.nii.gz` image is a different acquisition with a different
readout time and is not a matched two-blip partner; it is kept for reference only.

### Mechanics

- Integration tests live in `tests/` behind `UNPILE_FIXTURES` and `UNPILE_TESTDATA`; without the
  variables they skip.
- Assertions go through one helper, `expect(score, lo, hi, note)`, whose message names the quantity,
  its value and the bound, so the CI log reads without the code.
- Reference values are recorded in `tests/fixtures/bounds.toml` with a tolerance per metric, so a
  regression is reported against a number, not a rerun. First values (T2w fieldmap-less vs the SE
  TOPUP field): ds002737 SBRef 0.651 / 0.41 / 1.84 mm, BOLD5000 ses-01 0.863 / 0.66 / 1.54 mm.
- Local only: CRASH (not redistributable) and CSDSI2 (HBCD protocol; used for figures, not in the bundle).
