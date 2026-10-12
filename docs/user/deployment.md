# Deployment: containers, HPC and GPUs

## Shape of the artifact

unpile is one statically linked Rust binary (plus an optional PyO3 module later). Its only dynamic
dependencies are glibc, libm and libgcc_s; the GPU backends load their drivers at runtime, so one
build serves CPU-only and GPU machines:

- `--device flex` (default): pure Rust, SIMD dispatched at runtime (AVX2/AVX-512/NEON, scalar
  fallback), rayon threads sized by `--threads`. Needs nothing from the system.
- `--device wgpu` (feature `wgpu`): wgpu opens the Vulkan loader (`libvulkan.so.1`) with dlopen; if
  it is absent the device creation fails and nothing else is affected. Metal on macOS, Vulkan on
  Linux; the GL fallback is not useful for compute.
- `--device cuda` (feature `cuda`, not built by default): Burn's CUDA backend goes through `cudarc`
  with `fallback-dynamic-loading`, i.e. `libcuda.so` (driver) and `libnvrtc.so` (CUDA toolkit, for
  kernel JIT) are dlopen'ed on first use. The binary runs without CUDA present. The build pulls
  `cubecl-llvm`, which fetches a bundled LLVM, so it is a long build, and `libnvrtc.so` must ship in
  the image (it is not part of the driver). CUDA honours `CUDA_VISIBLE_DEVICES`; Vulkan does not.

## Distribution

The convenient route for the qsiprep image is a Python wheel built by maturin in `bin` mode
(`[tool.maturin] bindings = "bin"`): the binary is packaged as a console script and installed on
`PATH` by `pip`/`pixi`. Wheels: manylinux_2_28 x86_64 and aarch64, macOS arm64. The `itk` feature
(static HDF5 for Composite.h5 reading) is a build-time choice; the warp writer needs only `nifti`.
A conda-forge recipe can follow later if the pixi lock wants everything from conda channels.

## Containers and HPC

- **Docker with the NVIDIA runtime**: Vulkan needs `NVIDIA_DRIVER_CAPABILITIES=graphics` (or `all`);
  the default `compute,utility` mounts CUDA but not the Vulkan/GL driver libraries. CUDA needs only
  the default.
- **Apptainer `--nv`**: its `nvliblist.conf` binds `libcuda.so`, `libGLX_nvidia.so`,
  `libnvidia-glvkspirv.so`, `libnvidia-vulkan-producer.so` and the `nvidia_icd*.json` ICD files from
  the host, so both the wgpu and the CUDA backend see the GPU. The image itself must contain the
  Vulkan loader package (`libvulkan1`) for wgpu, and `libnvrtc.so` for CUDA.
- **Multi-GPU nodes**: a fit uses one GPU. SLURM hands out `CUDA_VISIBLE_DEVICES`, which the CUDA
  backend respects and Vulkan ignores (it enumerates every GPU on the node). For wgpu the device
  index therefore has to be chosen explicitly (`--device wgpu:N`, to add) or mapped from the PCI
  bus id that wgpu reports per adapter; until then CUDA is the safer backend on shared nodes.
- **CPU-only HPC**: the default device; set `--threads` to the allocated cores. Nothing to install.

## Build matrix

| features | runtime needs | when |
|---|---|---|
| (none) | glibc | CPU-only wheel, wasm library |
| `wgpu` | `libvulkan1` + a driver for GPU use; optional otherwise | the general wheel |
| `wgpu,cuda` | as above, plus `libnvrtc.so` in the image for CUDA | the qsiprep GPU image |
| `itk` | nothing extra (static HDF5 at build time) | CLI builds that read Composite.h5 |

## Binary size

Release build, `lto = "fat"`, `codegen-units = 1`, `strip = true` (the plain release build is 115 MB,
87 MB stripped):

| build | size |
|---|---|
| wgpu+itk | 64 MB |
| wgpu, no itk | 60 MB |
| CPU only | 10 MB |

Most of it is burn-flex's SIMD kernel variants and wgpu/naga; the static HDF5 of the `itk` feature adds about 4 MB.
