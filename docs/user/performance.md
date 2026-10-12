# Performance

unpile runs on the CPU by default and on a GPU when one is present. Nothing in the algorithm
depends on the device: the same fit on both gives the same numbers to float rounding.

## CPU

`--device flex` is a pure-Rust backend with SIMD kernels dispatched at run time (AVX2, AVX-512,
NEON) and a rayon thread pool sized with `--threads`; pipelines should pass their per-process
allocation there. Three design choices keep it fast: the smoothing kernel of the forward model is
a tent (the trilinear hat) whose membership function is a clamped quadratic, so the inner loop
has no transcendental functions (`--membership gauss` restores the prototype's Gaussian at two to
three times the cost for the same accuracy); the local-correlation window is three separable
one-dimensional filters; and the forward model is evaluated for all displacement offsets in one
vectorized pass.

Measured pinned to four cores, the size of a GitHub Actions runner:

| fit | time |
|---|---|
| 3 mm fixture, same-contrast reference, short schedule (levels 24/12/6, 20 iterations) | 25 s |
| 3 mm fixture, GRE prior with pose, shim and residual, short schedule | 37 s |
| 3 mm fixture, T2w with joint pose, short schedule | 41 s |
| 2.4 mm SBRef, T2w with joint pose, full schedule (levels 24/12/6/3, 60 iterations) | 6.3 min |
| the same on the 1.8 mm grid recommended in {doc}`fieldmapless` | 8 min (3 mm fixture at 2 mm) |

## GPU

`--device wgpu` runs the same code through WebGPU: Vulkan on Linux, Metal on macOS. The 2.4 mm
SBRef fit above takes 40 s, the 3 mm fixtures about 10 s each. On a node with several GPUs the
first discrete GPU is used; see {doc}`deployment` for the `CUDA_VISIBLE_DEVICES` caveat.

## Profiling

`UNPILE_PROFILE=1` prints inclusive wall time per operation at the end of a CPU run (forward
model, metric, smoothing, backward pass). It is how the choices above were made and the place to
look if a new input is slow.
