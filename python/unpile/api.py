# SPDX-License-Identifier: MIT OR Apache-2.0
"""A thin Python layer over the unpile binary.

`fit` runs one estimation and returns the output paths and scores. `prepare` puts the reference,
mask and fieldmap on the EPI grid, optionally a finer one, so that a BIDS-style set of files can be
handed over directly; it needs nibabel and nilearn (`pip install "unpile[prep]"`).
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._find import find_unpile_bin

PathLike = str | os.PathLike[str]


@dataclass
class FitResult:
    """Where a run wrote its files, and what it scored."""

    out_dir: Path
    field_hz: Path
    displacement_mm: Path
    corrected: Path
    sdc_warp: Path
    pose_xfm: Path | None
    fieldmap_pose_xfm: Path | None
    corrected_reverse: Path | None
    score: dict[str, Any] = field(default_factory=dict)
    log: str = ""


def sidecar_values(sidecar: PathLike | Mapping[str, Any]) -> tuple[str, float]:
    """`(PhaseEncodingDirection, TotalReadoutTime)` from a BIDS sidecar path or dict."""
    meta = sidecar if isinstance(sidecar, Mapping) else json.load(open(sidecar))
    try:
        return str(meta["PhaseEncodingDirection"]), float(meta["TotalReadoutTime"])
    except KeyError as e:
        raise KeyError(f"the sidecar lacks {e}; unpile needs PhaseEncodingDirection and TotalReadoutTime") from None


def run(args: Sequence[str], *, device: str = "flex", threads: int | None = None, membership: str | None = None, capture: bool = True) -> subprocess.CompletedProcess[str]:
    """Run the binary with global options first, then `args`."""
    cmd = [find_unpile_bin(), "--device", device]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    if membership is not None:
        cmd += ["--membership", membership]
    cmd += list(args)
    return subprocess.run(cmd, check=True, text=True, capture_output=capture)


def fit(
    epi: PathLike,
    mask: PathLike,
    out_dir: PathLike,
    *,
    pe_dir: str | None = None,
    total_readout_time: float | None = None,
    sidecar: PathLike | Mapping[str, Any] | None = None,
    reference: PathLike | None = None,
    reverse_epi: PathLike | None = None,
    fieldmap: PathLike | None = None,
    fieldmap_refine: str | None = None,
    metric: str | None = None,
    pose: bool = False,
    lam_mm: float | None = None,
    acq_voxel_mm: float | None = None,
    levels: Sequence[float] | None = None,
    iters: int | None = None,
    truth_field: PathLike | None = None,
    truth_xfm: PathLike | None = None,
    device: str = "flex",
    threads: int | None = None,
    extra: Sequence[str] = (),
) -> FitResult:
    """Estimate the field. Every image must be on the grid of `epi`; see `prepare`.

    `pe_dir` and `total_readout_time` can come from `sidecar` instead. With `reference` and
    `metric="lncc"` the reference is a T2w; with `reverse_epi` the second image has the opposite
    phase-encoding direction; `fieldmap` is a GRE field in Hz on the EPI grid.
    """
    if sidecar is not None:
        pe_dir, total_readout_time = sidecar_values(sidecar)
    if pe_dir is None or total_readout_time is None:
        raise ValueError("pass pe_dir and total_readout_time, or a sidecar that has them")
    out = Path(out_dir)
    args: list[str] = ["fit", "--epi", str(epi), "--mask", str(mask), "--out", str(out), "--pe-dir", pe_dir, "--total-readout-time", repr(float(total_readout_time))]
    for flag, value in (("--reference", reference), ("--reverse-epi", reverse_epi), ("--fieldmap", fieldmap), ("--fieldmap-refine", fieldmap_refine), ("--metric", metric), ("--truth-field", truth_field), ("--truth-xfm", truth_xfm)):
        if value is not None:
            args += [flag, str(value)]
    if pose:
        args.append("--pose")
    if lam_mm is not None:
        args += ["--lam-mm", repr(float(lam_mm))]
    if acq_voxel_mm is not None:
        args += ["--acq-voxel-mm", repr(float(acq_voxel_mm))]
    if levels is not None:
        args += ["--levels", ",".join(str(v) for v in levels)]
    if iters is not None:
        args += ["--iters", str(iters)]
    args += list(extra)
    proc = run(args, device=device, threads=threads)
    score_path = out / "score.json"
    score = json.load(open(score_path)) if score_path.exists() else {}
    opt = lambda name: (out / name) if (out / name).exists() else None  # noqa: E731
    return FitResult(
        out_dir=out,
        field_hz=out / "field_hz.nii.gz",
        displacement_mm=out / "disp_pe_mm.nii.gz",
        corrected=out / "corrected.nii.gz",
        sdc_warp=out / "sdc_warp.nii.gz",
        pose_xfm=opt("pose_xfm.txt"),
        fieldmap_pose_xfm=opt("gre_pose_xfm.txt"),
        corrected_reverse=opt("corrected_pa.nii.gz"),
        score=score,
        log=proc.stdout,
    )


def check(testdata_dir: PathLike, *, device: str = "flex") -> str:
    """Run the operator check against the stored PyTorch dumps; returns its report."""
    return run(["check", "--dir", str(testdata_dir)], device=device).stdout


def prepare(
    epi: PathLike,
    out_dir: PathLike,
    *,
    mask: PathLike | None = None,
    reference: PathLike | None = None,
    fieldmap: PathLike | None = None,
    reverse_epi: PathLike | None = None,
    grid_mm: float | None = None,
    pe_dir: str | None = None,
    sidecar: PathLike | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resample everything onto the EPI grid, or onto a finer grid of `grid_mm` voxels.

    A finer grid keeps the detail of a T2w reference; two thirds of the EPI voxel is the usual
    choice. Returns the new paths plus `acq_voxel_mm` (the acquired phase-encoding voxel size, which
    `fit` needs on a resampled grid). Needs nibabel and nilearn.
    """
    import nibabel as nb
    import numpy as np
    from nilearn.image import resample_img, resample_to_img

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    epi_img = nb.load(str(epi))
    if pe_dir is None and sidecar is not None:
        pe_dir, _ = sidecar_values(sidecar)
    axis = "ijk".index(pe_dir[0]) if pe_dir else 1
    acq_voxel = float(epi_img.header.get_zooms()[axis])
    if grid_mm is not None and abs(grid_mm - acq_voxel) > 1e-3:
        scale = np.array([grid_mm / z for z in epi_img.header.get_zooms()[:3]])
        affine = epi_img.affine.copy()
        affine[:3, :3] = epi_img.affine[:3, :3] * scale
        shape = tuple(int(np.ceil(s / sc)) for s, sc in zip(epi_img.shape[:3], scale))
        target = resample_img(epi_img, target_affine=affine, target_shape=shape, interpolation="continuous")
    else:
        target = epi_img
    paths: dict[str, Any] = {"acq_voxel_mm": acq_voxel}
    nb.save(target, str(out / "epi.nii.gz"))
    paths["epi"] = out / "epi.nii.gz"
    for name, src, interp in (("mask", mask, "nearest"), ("reference", reference, "continuous"), ("fieldmap", fieldmap, "continuous"), ("reverse_epi", reverse_epi, "continuous")):
        if src is None:
            continue
        img = resample_to_img(nb.load(str(src)), target, interpolation=interp)
        nb.save(img, str(out / f"{name}.nii.gz"))
        paths[name] = out / f"{name}.nii.gz"
    return paths
