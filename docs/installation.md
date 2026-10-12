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

There is no Python module. qsiprep calls the binary as a subprocess. The commands in
{doc}`quickstart` are what that call looks like.
