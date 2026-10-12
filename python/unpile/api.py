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
    try:
        warn_if_not_las(epi)
    except ImportError:
        pass
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


def reorient_pe_dir(pe_dir: str, source: PathLike | Any, target: PathLike | Any) -> str:
    """Translate a sidecar PhaseEncodingDirection from the grid it was written for to another grid.

    `source` is the image (or affine) the sidecar describes, `target` the image (or affine) you are
    about to fit; both are NIfTI paths, nibabel images or 4x4 affines. The voxel axes may have been
    permuted or flipped between them (qsiprep reorients to LAS+, for example); the direction is
    carried through that change so that `j-` on the original file becomes what it is on the new one.
    Only the orientation matters: a resampling that keeps the axes pointing the same way leaves the
    direction unchanged.
    """
    import nibabel as nb
    import numpy as np

    def affine_of(x):
        if isinstance(x, (str, os.PathLike)):
            return nb.load(str(x)).affine
        return x.affine if hasattr(x, "affine") else np.asarray(x, float)

    src_ornt = nb.orientations.io_orientation(affine_of(source))  # voxel axis -> (world axis, direction)
    tgt_ornt = nb.orientations.io_orientation(affine_of(target))
    axis = "ijk".index(pe_dir[0])
    sign = -1.0 if pe_dir.endswith("-") else 1.0
    world_axis, world_dir = int(src_ornt[axis, 0]), float(src_ornt[axis, 1]) * sign  # the PE direction in world terms
    for new_axis in range(3):
        if int(tgt_ornt[new_axis, 0]) == world_axis:
            new_sign = world_dir * float(tgt_ornt[new_axis, 1])
            return "ijk"[new_axis] + ("-" if new_sign < 0 else "")
    raise ValueError("the two images do not share a world axis for the phase-encoding direction")


def check_polarity(
    epi: PathLike,
    mask: PathLike,
    reference: PathLike,
    fieldmap: PathLike,
    pe_dir: str,
    total_readout_time: float,
    *,
    metric: str = "ssd",
    acq_voxel_mm: float | None = None,
    work_dir: PathLike | None = None,
    device: str = "flex",
    threads: int | None = None,
    warn: bool = True,
) -> dict[str, Any]:
    """Check the sidecar polarity against a fieldmap: apply the fieldmap as given under `pe_dir` and
    under the opposite polarity, and compare how well each explains the EPI.

    Under the right polarity the fieldmap unwarps the EPI toward the reference; under the wrong one
    it doubles the distortion, so the data terms differ by a wide margin. A reference alone cannot
    decide the sign: with a free field, the opposite polarity fits the negated field and reaches
    the same data term. Returns both data terms, the better polarity and the margin, and warns when
    the better one is not `pe_dir`.
    """
    import tempfile
    import warnings

    axis, flipped = pe_dir[0], pe_dir.endswith("-")
    other = axis if flipped else axis + "-"
    work = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="unpile_polarity_"))
    terms = {}
    for d in (pe_dir, other):
        r = fit(epi, mask, work / d.replace("-", "neg"), pe_dir=d, total_readout_time=total_readout_time, reference=reference, metric=metric,
                fieldmap=fieldmap, fieldmap_refine="none", acq_voxel_mm=acq_voxel_mm, levels=[24], iters=1, device=device, threads=threads)
        terms[d] = float(r.score["data_term"]["initial"])
    better = min(terms, key=terms.get)
    worst = max(terms.values())
    margin = (worst - terms[better]) / worst if worst > 0 else 0.0
    result = {"given": pe_dir, "better": better, "data_term": terms, "margin": margin, "consistent": better == pe_dir}
    if warn and better != pe_dir:
        warnings.warn(f"the fieldmap explains the EPI better under the opposite polarity ({better}) than under the sidecar's ({pe_dir}): data term {terms[better]:.4f} vs {terms[pe_dir]:.4f}; check the image orientation against the sidecar, and the fieldmap's sign", stacklevel=2)
    return result


def axcodes(image: PathLike | Any) -> tuple[str, str, str]:
    """The world direction of each voxel axis, e.g. ('L', 'A', 'S')."""
    import nibabel as nb

    img = nb.load(str(image)) if isinstance(image, (str, os.PathLike)) else image
    return tuple(nb.aff2axcodes(img.affine))


def warn_if_not_las(image: PathLike | Any, what: str = "the EPI") -> bool:
    """Warn when an image is not LAS+, the orientation qsiprep's tools work in; returns whether it is."""
    import warnings

    codes = axcodes(image)
    if codes != ("L", "A", "S"):
        warnings.warn(f"{what} is {''.join(codes)}+, not LAS+; qsiprep and the tools it combines unpile with work in LAS+, so reorient first (unpile.prepare does) and carry the phase-encoding direction through the reorientation (unpile.reorient_pe_dir)", stacklevel=3)
        return False
    return True


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
    orientation: str | None = "LAS",
) -> dict[str, Any]:
    """Put everything on the EPI grid, reoriented to LAS+ unless `orientation` is None, optionally
    on a finer grid of `grid_mm` voxels.

    A finer grid keeps the detail of a T2w reference; two thirds of the EPI voxel is the usual
    choice. Returns the new paths, `pe_dir` carried through the reorientation, and `acq_voxel_mm`
    (the acquired phase-encoding voxel size, which `fit` needs on a resampled grid). Needs nibabel
    and nilearn.
    """
    import nibabel as nb
    import numpy as np
    from nilearn.image import resample_img, resample_to_img

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    epi_img = nb.load(str(epi))
    if pe_dir is None and sidecar is not None:
        pe_dir, _ = sidecar_values(sidecar)
    if orientation is not None and tuple(nb.aff2axcodes(epi_img.affine)) != tuple(orientation):
        src = epi_img
        ornt = nb.orientations.ornt_transform(nb.orientations.io_orientation(src.affine), nb.orientations.axcodes2ornt(tuple(orientation)))
        epi_img = src.as_reoriented(ornt)
        if pe_dir:
            pe_dir = reorient_pe_dir(pe_dir, src, epi_img)
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
    paths: dict[str, Any] = {"acq_voxel_mm": acq_voxel, "pe_dir": pe_dir}
    nb.save(target, str(out / "epi.nii.gz"))
    paths["epi"] = out / "epi.nii.gz"
    for name, src, interp in (("mask", mask, "nearest"), ("reference", reference, "continuous"), ("fieldmap", fieldmap, "continuous"), ("reverse_epi", reverse_epi, "continuous")):
        if src is None:
            continue
        img = resample_to_img(nb.load(str(src)), target, interpolation=interp)
        nb.save(img, str(out / f"{name}.nii.gz"))
        paths[name] = out / f"{name}.nii.gz"
    return paths
