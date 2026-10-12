# Command line

One binary, three subcommands. The global options `--device`, `--threads` and `--membership` go before the subcommand.

```text
Susceptibility distortion correction for EPI: refine a measured GRE fieldmap against the EPI, or estimate the field from an anatomical reference (fieldmap-less); opposite-polarity pairs are also supported

Usage: unpile [OPTIONS] <COMMAND>

Commands:
  fit     Estimate the field: one EPI against a reference (optionally with a GRE prior), or two EPIs of opposite polarity
  export  Write a field (and optionally a pose) as ITK transforms, with the internally resampled images for comparison
  check   Compare the operators with the PyTorch reference dumps in a testdata directory
  help    Print this message or the help of the given subcommand(s)

Options:
      --device <DEVICE>          compute device: flex (pure-Rust CPU) or wgpu (Vulkan/Metal/WebGPU, needs the `wgpu` feature) [default: flex]
      --threads <THREADS>        CPU threads for the flex backend (default: all cores; qsiprep passes its per-process allocation)
      --membership <MEMBERSHIP>  voxel membership kernel of the pushforward: tent (trilinear hat, fast) or gauss (the prototype's) [default: tent]
  -h, --help                     Print help
```

```{toctree}
:maxdepth: 1

fit
export
check
```
