# Deployment

## What is shipped

unpile is one statically linked binary. Its only dynamic dependencies are glibc, libm and libgcc_s.
The GPU backends load their drivers at run time, so one build serves CPU-only and GPU machines:

- `--device flex`, the default: pure Rust, nothing needed from the system.
- `--device wgpu`: opens the Vulkan loader (`libvulkan.so.1`) or Metal with dlopen. If it is absent
  the device fails to open and nothing else is affected.
- `--device cuda` (feature `cuda`, not built by default): loads `libcuda.so` and `libnvrtc.so` on first
  use. The binary runs without CUDA present. `libnvrtc.so` must be in the image; it is not part of
  the driver. The build fetches a bundled LLVM and is slow.

## The wheel

The PyPI wheel is built by maturin in `bin` mode: the binary is installed as a console script.
Wheels exist for manylinux x86_64 and macOS arm64. Sizes with LTO and stripping:

| build | size |
|---|---|
| wgpu and the ITK reader | 64 MB |
| wgpu only | 60 MB |
| CPU only | 10 MB |

## Containers and HPC

- Docker with the NVIDIA runtime: Vulkan needs `NVIDIA_DRIVER_CAPABILITIES=graphics` or `all`. The
  default `compute,utility` mounts CUDA but not the Vulkan driver libraries. CUDA needs only the default.
- Apptainer `--nv` binds `libcuda.so`, the NVIDIA Vulkan libraries and the `nvidia_icd*.json` files
  from the host. The image needs the Vulkan loader package (`libvulkan1`) for wgpu, and `libnvrtc.so`
  for CUDA.
- Several GPUs on a node: a fit uses one. SLURM sets `CUDA_VISIBLE_DEVICES`, which CUDA respects and
  Vulkan ignores; Vulkan lists every GPU on the node. For wgpu on a shared node the device index has
  to be chosen explicitly (`--device wgpu:N`, planned). CUDA is the safer backend there.
- CPU-only nodes: the default device. Set `--threads` to the allocated cores.

## Build matrix

| features | runtime needs | use |
|---|---|---|
| none | glibc | CPU-only wheel, WebAssembly library |
| `wgpu` | `libvulkan1` and a driver for GPU use; optional otherwise | the general wheel |
| `wgpu,cuda` | as above, plus `libnvrtc.so` in the image | a GPU image |
| `itk` (default) | nothing extra; static HDF5 at build time | reading `Composite.h5` transforms |
