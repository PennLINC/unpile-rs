# unpile export

Write an existing field (and a pose) as ITK transforms, with the internally resampled images for comparison against `antsApplyTransforms`. See {doc}`../user/inputs-outputs`.

```text
Write a field (and optionally a pose) as ITK transforms, with the internally resampled images for comparison

Usage: unpile export [OPTIONS] --field <FIELD> --out <OUT> --trt <TRT>

Options:
      --device <DEVICE>          compute device: flex (pure-Rust CPU) or wgpu (Vulkan/Metal/WebGPU, needs the `wgpu` feature) [default: flex]
      --field <FIELD>            field in Hz on the EPI grid (the file's own axis order)
      --obs <OBS>                distorted image to pull back along the field (writes pullback.nii.gz)
      --threads <THREADS>        CPU threads for the flex backend (default: all cores; qsiprep passes its per-process allocation)
      --membership <MEMBERSHIP>  voxel membership kernel of the pushforward: tent (trilinear hat, fast) or gauss (the prototype's) [default: tent]
      --reference <REFERENCE>    reference image to move by --pose (writes posed.nii.gz)
      --pose <POSE>              six pose parameters as the optimizer reports them (rx,ry,rz,tx,ty,tz in POSE_ROT/POSE_TRANS units)
      --out <OUT>                output directory
      --trt <TRT>                TotalReadoutTime of the EPI in seconds
      --pe-axis <PE_AXIS>        voxel axis of phase encoding [default: 1]
      --sign <SIGN>              polarity, as for fit [default: -1]
  -h, --help                     Print help
```
