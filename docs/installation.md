# Installation

## From PyPI

```bash
pip install unpile
```

The wheel contains one binary, `unpile`. It runs on the CPU by default and needs nothing else. With
a GPU it uses Vulkan on Linux or Metal on macOS; {doc}`user/deployment` lists what a container needs
for that.

## From source

```bash
git clone https://github.com/PennLINC/unpile-rs
cd unpile-rs
cargo build --release --features wgpu
./target/release/unpile --help
```

The Rust toolchain is pinned by `rust-toolchain.toml`. Without `--features wgpu` the binary is
CPU-only and 10 MB instead of 60 MB.

## Python

The wheel also installs a small `unpile` Python package that finds the binary and wraps it:

```python
import unpile

r = unpile.fit("se_ap.nii.gz", "brain_mask.nii.gz", "out", sidecar="se_ap.json",
               reference="t2w.nii.gz", metric="lncc", pose=True, lam_mm=60)
r.sdc_warp, r.corrected, r.score
```

`unpile.prepare` resamples a reference, mask and fieldmap onto the EPI grid or onto a finer one. It
requires the EPI to be LAS+ and refuses otherwise; `unpile.reorient` does the reorientation when
asked and carries the phase-encoding direction through it. Both need `pip install "unpile[prep]"`
(nibabel and nilearn). `python -m unpile ...` runs the binary
with the same arguments as the console script. qsiprep uses this package from its own interface.
