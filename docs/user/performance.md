# Performance

unpile runs on the CPU by default and on a GPU when one is present. The results are the same on
both to float rounding.

## CPU

`--device flex` is a pure-Rust backend with SIMD kernels chosen at run time (AVX2, AVX-512, NEON)
and a thread pool sized with `--threads`. Pipelines should pass their per-process thread count there.

Three choices keep it fast. The smoothing kernel of the forward model is a tent, so the inner loop
has no transcendental functions; `--membership gauss` restores the Gaussian kernel of the prototype
at two to three times the cost for the same results. The local-correlation window is three separable
one-dimensional filters. The forward model handles all displacement offsets in one vectorized pass.

Timings on four cores:

| fit | time |
|---|---|
| 3 mm simulated data, same-contrast reference, short schedule (levels 24/12/6 mm, 20 iterations) | 25 s |
| 3 mm simulated data, fieldmap with pose, shim and residual, short schedule | 37 s |
| 3 mm simulated data, T2w with pose, short schedule | 41 s |
| 2.4 mm SBRef, T2w with pose, full schedule (levels 24/12/6/3 mm, 60 iterations) | 6.3 min |
| 3 mm simulated data fitted on a 2 mm grid, T2w with pose, full schedule | 8 min |

## GPU

`--device wgpu` runs the same code through Vulkan on Linux or Metal on macOS. The 2.4 mm SBRef fit
takes 40 s and the 3 mm fits about 10 s each. On a node with several GPUs the first discrete GPU
is used; see {doc}`deployment`.

## Profiling

`UNPILE_PROFILE=1` prints the wall time per operation at the end of a CPU run.
