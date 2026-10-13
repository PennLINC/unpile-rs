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
    reference_posed: Path | None = None
    pose_composed_xfm: Path | None = None
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
    truth_fieldmap_xfm: PathLike | None = None,
    reference_init_xfm: PathLike | None = None,
    device: str = "flex",
    threads: int | None = None,
    extra: Sequence[str] = (),
) -> FitResult:
    """Estimate the field. Every image must be on the grid of `epi`; see `prepare`.

    `pe_dir` and `total_readout_time` can come from `sidecar` instead. With `reference` and
    `metric="lncc"` the reference is a T2w; with `reverse_epi` the second image has the opposite
    phase-encoding direction; `fieldmap` is a GRE field in Hz on the EPI grid.
    `reference_init_xfm` is the transform `prepare` applied to the reference (`reference_xfm` in its
    result); with it the run also writes `pose_composed_xfm.txt`, from the EPI grid to the original
    reference.
    """
    if sidecar is not None:
        pe_dir, total_readout_time = sidecar_values(sidecar)
    if pe_dir is None or total_readout_time is None:
        raise ValueError("pass pe_dir and total_readout_time, or a sidecar that has them")
    try:
        warn_if_not_las(epi)
        preflight(epi, mask, reference=reference, fieldmap=fieldmap, reverse_epi=reverse_epi)
    except ImportError:
        pass
    out = Path(out_dir)
    args: list[str] = ["fit", "--epi", str(epi), "--mask", str(mask), "--out", str(out), "--pe-dir", pe_dir, "--total-readout-time", repr(float(total_readout_time))]
    for flag, value in (("--reference", reference), ("--reverse-epi", reverse_epi), ("--fieldmap", fieldmap), ("--fieldmap-refine", fieldmap_refine), ("--metric", metric), ("--truth-field", truth_field), ("--truth-xfm", truth_xfm), ("--truth-fieldmap-xfm", truth_fieldmap_xfm), ("--reference-init-xfm", reference_init_xfm)):
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
        reference_posed=opt("reference_posed.nii.gz"),
        pose_composed_xfm=opt("pose_composed_xfm.txt"),
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


def reorient(image: PathLike, orientation: str = "LAS", *, pe_dir: str | None = None, out: PathLike | None = None) -> tuple[Path, str | None]:
    """Write `image` with its voxel axes permuted and flipped to `orientation` (e.g. "LAS"), and
    return the new path with `pe_dir` carried through the change.

    Nothing is resampled: the voxels are the same, only their order changes. This is the explicit
    step `prepare` asks for when an EPI is not LAS+; the returned direction is the one to pass to
    `fit`, and it may differ from the sidecar's.
    """
    import nibabel as nb

    src = nb.load(str(image))
    target = tuple(orientation)
    if tuple(nb.aff2axcodes(src.affine)) == target:
        return Path(image), pe_dir
    ornt = nb.orientations.ornt_transform(nb.orientations.io_orientation(src.affine), nb.orientations.axcodes2ornt(target))
    dst = src.as_reoriented(ornt)
    out_path = Path(out) if out is not None else Path(image).with_name(Path(image).name.replace(".nii", f"_{orientation}.nii", 1))
    nb.save(dst, str(out_path))
    new_dir = reorient_pe_dir(pe_dir, src, dst) if pe_dir else None
    return out_path, new_dir


def preflight(
    epi: PathLike,
    mask: PathLike,
    *,
    reference: PathLike | None = None,
    fieldmap: PathLike | None = None,
    reverse_epi: PathLike | None = None,
    max_com_offset_mm: float = 10.0,
    min_coverage: float = 0.95,
    warn: bool = True,
) -> dict[str, Any]:
    """Sanity checks before a fit, for the mistakes the binary cannot see.

    Reports, per input: that it is on the EPI grid; its coverage of the brain mask (the fraction of
    mask voxels where it has signal; a fieldmap or anatomical whose field of view stops short of the
    brain shows up here); and, for the reference and the reverse-polarity image, the distance
    between its centre of mass and the EPI's inside the mask, which is large when the reference was
    never registered or is the wrong file. Also the mask volume and NaNs. Warnings by default;
    the returned dict has the numbers. Needs nibabel.
    """
    import warnings

    import nibabel as nb
    import numpy as np

    def load(p):
        img = nb.load(str(p))
        return img, np.asarray(img.dataobj, np.float32).squeeze()

    def com(data, weights_mask, affine):
        w = np.where(weights_mask, np.clip(data, 0, None), 0.0)
        if w.sum() <= 0:
            return None
        idx = np.array(np.nonzero(weights_mask), float)
        c = (idx * w[weights_mask]).sum(axis=1) / w[weights_mask].sum()
        return (affine @ np.append(c, 1.0))[:3]

    epi_img, epi_d = load(epi)
    mask_img, mask_d = load(mask)
    m = mask_d > 0.5
    report: dict[str, Any] = {"warnings": []}

    def warn_(msg):
        report["warnings"].append(msg)
        if warn:
            warnings.warn(msg, stacklevel=3)

    vol_cm3 = float(m.sum() * np.prod(mask_img.header.get_zooms()[:3]) / 1000.0)
    report["mask_volume_cm3"] = vol_cm3
    if m.sum() == 0:
        raise ValueError("the brain mask is empty")
    if not 400 <= vol_cm3 <= 2200:
        warn_(f"the brain mask is {vol_cm3:.0f} cm3, outside the 400-2200 cm3 a head should give; check it is a brain mask on the EPI grid")
    if not np.isfinite(epi_d).all():
        warn_("the EPI contains NaN or infinite values")
    epi_com = com(epi_d, m, epi_img.affine)
    report["epi_com_mm"] = None if epi_com is None else [float(v) for v in epi_com]
    for name, path in (("reference", reference), ("fieldmap", fieldmap), ("reverse_epi", reverse_epi)):
        if path is None:
            continue
        img, d = load(path)
        entry: dict[str, Any] = {}
        same_grid = img.shape[:3] == epi_img.shape[:3] and np.allclose(img.affine, epi_img.affine, atol=1e-2)
        entry["on_epi_grid"] = bool(same_grid)
        if not same_grid:
            warn_(f"{name} is not on the EPI grid (shape {img.shape[:3]} vs {epi_img.shape[:3]}); unpile.prepare puts it there")
            report[name] = entry
            continue
        finite = np.isfinite(d)
        if not finite.all():
            warn_(f"{name} contains NaN or infinite values")
        signal = finite & (np.abs(d) > 0)
        coverage = float(signal[m].mean())
        entry["coverage_of_mask"] = coverage
        if coverage < min_coverage:
            warn_(f"{name} has signal in only {100 * coverage:.0f}% of the brain mask; its field of view may stop short of the brain")
        if name in ("reference", "reverse_epi"):
            # the input's own signal within a generous neighbourhood of the brain, so that a shifted
            # or wrong image moves the centre of mass instead of being clipped by the mask
            from scipy import ndimage as ndi

            near = ndi.binary_dilation(m, iterations=8)
            strong = finite & (d > 0.15 * np.percentile(d[m & finite], 99)) if (m & finite).any() else finite
            c = com(d, near & strong, img.affine)
            if c is not None and epi_com is not None:
                off = float(np.linalg.norm(c - epi_com))
                entry["com_offset_mm"] = off
                if off > max_com_offset_mm:
                    warn_(f"{name}'s centre of mass is {off:.1f} mm from the EPI's inside the brain mask; a registered image should be within a few mm, so check that it is the right file and was registered to the EPI")
        if name == "fieldmap":
            rms = float(np.sqrt(np.mean(d[m & finite] ** 2)))
            entry["rms_hz"] = rms
            if not 1.0 <= rms <= 300.0:
                warn_(f"the fieldmap's rms inside the brain is {rms:.1f} Hz; a field in Hz is usually 10-100 Hz rms, so check its units")
        report[name] = entry
    return report


def check(testdata_dir: PathLike, *, device: str = "flex") -> str:
    """Run the operator check against the stored PyTorch dumps; returns its report."""
    return run(["check", "--dir", str(testdata_dir)], device=device).stdout


_LPS = (-1.0, -1.0, 1.0)


def read_itk_affine(path: PathLike) -> Any:
    """An ITK affine (.txt Insight Transform File or ANTs .mat) as a 4x4 RAS point map.

    ITK's convention: the map takes a point of the fixed image to the point it is read from in the
    moving image, the direction `antsApplyTransforms -t` uses. Rigid and affine transforms of the
    AffineTransform / MatrixOffsetTransformBase kind are read; anything else raises.
    """
    import numpy as np

    path = Path(path)
    if path.suffix == ".mat":
        from scipy.io import loadmat

        mat = loadmat(str(path))
        keys = [k for k in mat if k.startswith(("AffineTransform_", "MatrixOffsetTransformBase_"))]
        if len(keys) != 1:
            raise ValueError(f"{path}: expected one affine transform, found {sorted(k for k in mat if not k.startswith('__'))}")
        params = np.asarray(mat[keys[0]], float).ravel()
        fixed = np.asarray(mat.get("fixed", np.zeros(3)), float).ravel()
    else:
        kinds, params, fixed = [], None, np.zeros(3)
        for line in Path(path).read_text().splitlines():
            if line.startswith("Transform:"):
                kinds.append(line.split(":", 1)[1].strip())
            elif line.startswith("Parameters:"):
                params = np.array(line.split(":", 1)[1].split(), float)
            elif line.startswith("FixedParameters:"):
                fixed = np.array(line.split(":", 1)[1].split(), float)
        if len(kinds) != 1 or not kinds[0].startswith(("AffineTransform_", "MatrixOffsetTransformBase_")) or not kinds[0].endswith("_3_3"):
            raise ValueError(f"{path}: expected one 3-D AffineTransform, found {kinds}")
    if params is None or params.size != 12 or fixed.size != 3:
        raise ValueError(f"{path}: expected 12 parameters and a 3-D centre")
    m, t, c = params[:9].reshape(3, 3), params[9:], fixed
    lps = np.eye(4)
    lps[:3, :3] = m
    lps[:3, 3] = t + c - m @ c
    flip = np.diag(_LPS + (1.0,))
    return flip @ lps @ flip


def write_itk_affine(path: PathLike, ras: Any) -> Path:
    """Write a 4x4 RAS point map as an Insight Transform File (LPS, centre 0)."""
    import numpy as np

    flip = np.diag(_LPS + (1.0,))
    lps = flip @ np.asarray(ras, float) @ flip
    values = list(lps[:3, :3].ravel()) + list(lps[:3, 3])
    Path(path).write_text(
        "#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\n"
        "Parameters: " + " ".join(f"{v:.10f}" for v in values) + "\nFixedParameters: 0 0 0\n"
    )
    return Path(path)


def prepare(
    epi: PathLike,
    out_dir: PathLike,
    *,
    mask: PathLike | None = None,
    reference: PathLike | None = None,
    fieldmap: PathLike | None = None,
    reverse_epi: PathLike | None = None,
    reference_xfm: PathLike | None = None,
    grid_mm: float | None = None,
    pe_dir: str | None = None,
    sidecar: PathLike | Mapping[str, Any] | None = None,
    orientation: str | None = "LAS",
) -> dict[str, Any]:
    """Put everything on the EPI grid, optionally on a finer grid of `grid_mm` voxels.

    The EPI must be in `orientation` (LAS+ by default, the orientation qsiprep's tools work in);
    otherwise this raises and names `reorient`, which does the change explicitly. A finer grid keeps
    the detail of a T2w reference; two thirds of the EPI voxel is the usual choice. Returns the new
    paths, `pe_dir` and `acq_voxel_mm` (the acquired phase-encoding voxel size, which `fit` needs on
    a resampled grid). Needs nibabel and nilearn.

    `reference_xfm` is a rigid or affine ITK transform (.txt or ANTs .mat) from the EPI to the
    reference, in the direction `antsApplyTransforms -t` takes: for example the transform
    `antsRegistration` writes with the b=0 as the fixed image and the reference as the moving one. The reference is resampled through it, and a copy is returned as
    `reference_xfm` (written as `reference_xfm.txt`) to pass to `fit(reference_init_xfm=...)`.
    Without it the reference is placed by its header alone.
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
        found = "".join(nb.aff2axcodes(epi_img.affine))
        raise ValueError(
            f"the EPI is {found}+, not {orientation}+. unpile does not reorient images on its own, because a "
            f"reorientation changes which voxel axis is phase-encode. Reorient explicitly first, e.g. "
            f"epi, pe_dir = unpile.reorient(epi, {orientation!r}, pe_dir=pe_dir, out=...), which also carries "
            f"the phase-encoding direction through the change; or pass orientation=None to skip this check."
        )
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
    # field-of-view overlap: how much of the EPI's brain (or, without a mask, its volume) each input covers
    import warnings

    if mask is not None:
        mk = nb.load(str(mask))
        inside = np.array(np.nonzero(np.asarray(mk.dataobj) > 0.5), float)
        world = mk.affine @ np.vstack([inside, np.ones(inside.shape[1])])
    else:
        corners = np.array(np.meshgrid(*[[0, n - 1] for n in epi_img.shape[:3]], indexing="ij")).reshape(3, -1).astype(float)
        world = epi_img.affine @ np.vstack([corners, np.ones(corners.shape[1])])
    def placed(name, src):
        """The input image, with the reference moved by `reference_xfm` (by its affine, no resampling)."""
        img = nb.load(str(src))
        if name == "reference" and reference_xfm is not None:
            img = nb.Nifti1Image(np.asanyarray(img.dataobj), np.linalg.inv(read_itk_affine(reference_xfm)) @ img.affine, img.header)
        return img

    if reference_xfm is not None:
        if reference is None:
            raise ValueError("reference_xfm needs a reference")
        paths["reference_xfm"] = write_itk_affine(out / "reference_xfm.txt", read_itk_affine(reference_xfm))
    for name, src in (("mask", mask), ("reference", reference), ("fieldmap", fieldmap), ("reverse_epi", reverse_epi)):
        if src is None or name == "mask":
            continue
        img = placed(name, src)
        vox = np.linalg.inv(img.affine) @ world
        ok = np.all((vox[:3] >= -0.5) & (vox[:3] <= np.array(img.shape[:3])[:, None] - 0.5), axis=0)
        overlap = float(ok.mean())
        paths[f"{name}_fov_overlap"] = overlap
        if overlap < 0.5:
            raise ValueError(f"{name}'s field of view contains only {100 * overlap:.0f}% of the EPI's brain; check that it is the right file and that its header is right")
        if overlap < 0.95:
            warnings.warn(f"{name}'s field of view contains only {100 * overlap:.0f}% of the EPI's brain", stacklevel=2)
    for name, src, interp in (("mask", mask, "nearest"), ("reference", reference, "continuous"), ("fieldmap", fieldmap, "continuous"), ("reverse_epi", reverse_epi, "continuous")):
        if src is None:
            continue
        img = resample_to_img(placed(name, src), target, interpolation=interp)
        nb.save(img, str(out / f"{name}.nii.gz"))
        paths[name] = out / f"{name}.nii.gz"
    return paths
