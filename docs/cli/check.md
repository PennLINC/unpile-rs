# unpile check

Compare the operators with the stored PyTorch reference dumps (`testdata/rpe`). See {doc}`../user/validation`.

```text
Compare the operators with the PyTorch reference dumps in a testdata directory

Usage: unpile check [OPTIONS] --dir <DIR>

Options:
      --device <DEVICE>          compute device: flex (pure-Rust CPU) or wgpu (Vulkan/Metal/WebGPU, needs the `wgpu` feature) [default: flex]
      --dir <DIR>                directory with the dumps (testdata/rpe in the repository)
      --threads <THREADS>        CPU threads for the flex backend (default: all cores; qsiprep passes its per-process allocation)
      --membership <MEMBERSHIP>  voxel membership kernel of the pushforward: tent (trilinear hat, fast) or gauss (the prototype's) [default: tent]
  -h, --help                     Print help
```
