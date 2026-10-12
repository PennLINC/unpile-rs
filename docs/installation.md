# Installation

## From PyPI

```bash
pip install unpile
```

The wheel contains one statically linked binary, `unpile`, with the CPU backend (the default) and
the wgpu GPU backend. It needs nothing else on a CPU-only machine. For GPU use it loads the
system's Vulkan loader (Linux) or Metal (macOS) at run time; see {doc}`user/deployment` for what
containers and HPC nodes need.

## From source

```bash
git clone https://github.com/PennLINC/unpile-rs
cd unpile-rs
cargo build --release --features wgpu
./target/release/unpile --help
```

The toolchain is pinned by `rust-toolchain.toml`. Leave out `--features wgpu` for a CPU-only
binary (10 MB instead of 60 MB). The `itk` feature, on by default, adds the reader for ITK
`Composite.h5` and `.txt` transforms used by `--truth-xfm` and the pose output.

## Python

There is no Python module yet. qsiprep and other pipelines call the binary as a subprocess; the
command lines in {doc}`quickstart` are what such a call looks like.
