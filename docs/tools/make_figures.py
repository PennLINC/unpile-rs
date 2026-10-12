#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate the figures in docs/_static from fixture and test-data fits.

    python docs/tools/make_figures.py --fixtures testdata/fixtures --testdata-real <dir with ds002737 at 1.8 mm>
                                      --real-out <out dir of the 1.8 mm ds002737 fit>

The fixture panels use the `phasediff` and `rpe` fixtures after `cargo test --test fixtures` has
written their `unpile/out/*` fits. The SBRef panels use the ds002737 inputs resampled to 1.8 mm
(the two-thirds-voxel grid) and the fit written from them; see docs/user/fieldmapless.md.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np

GRAY = dict(cmap="gray")
FIELD = "RdBu_r"


def vol(path):
    img = nb.load(path)
    return np.asarray(img.dataobj, np.float32).squeeze(), img


def axial(a, z):
    """A slice in display orientation: anterior up (RAS/LAS arrays: y increases to anterior)."""
    return a[:, :, z].T


def centre_slice(mask):
    zs = np.where(mask.any(axis=(0, 1)))[0]
    return int(np.round(zs.mean()))


def crop(a2d, mask2d, margin=4):
    """Crop a display slice to the mask's bounding box (plus a margin) to drop the empty field of view."""
    ys, xs = np.where(mask2d)
    if len(ys) == 0:
        return a2d
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, a2d.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, a2d.shape[1])
    return a2d[y0:y1, x0:x1]


def masked(field, mask):
    out = field.copy()
    out[~mask] = np.nan
    return out


def panel(ax, a, title, mask=None, hz=None, aspect=1.0):
    if hz is None:
        lo, hi = np.percentile(a[np.isfinite(a)], [1, 99.5])
        ax.imshow(a, origin="lower", vmin=lo, vmax=hi, aspect=aspect, **GRAY)
    else:
        im = ax.imshow(a, origin="lower", cmap=FIELD, vmin=-hz, vmax=hz, aspect=aspect)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    return im if hz is not None else None


def gre_refinement(fx, out):
    d = os.path.join(fx, "phasediff", "unpile")
    gre, img = vol(f"{d}/gre_hz.nii.gz")
    truth, _ = vol(f"{d}/truth_field.nii.gz")
    refined, _ = vol(f"{d}/out/gre/field_hz.nii.gz")
    mask = vol(f"{d}/brain_mask.nii.gz")[0] > 0
    score = json.load(open(f"{d}/out/gre/score.json"))
    z = centre_slice(mask)
    aspect = img.header.get_zooms()[1] / img.header.get_zooms()[0]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    hz = 60
    m2 = axial(mask, z)
    for ax, a, t in zip(axes, [gre, refined, truth, refined - truth],
                        ["GRE as acquired\n(head moved 5.4° / 4 mm after the EPI)", "refined by unpile\n(pose + shim + residual)",
                         "true field", "refined − true"]):
        im = panel(ax, crop(axial(masked(a, mask), z), m2), t, hz=hz, aspect=aspect)
    cb = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.02)
    cb.set_label("Hz", fontsize=8)
    p = score["gre_pose_vs_truth"]
    fig.suptitle(f"GRE refinement on the TRXScan phasediff fixture (3 mm): pose residual "
                 f"{p['rotation_deg']:.2f}° / {p['translation_mm']:.2f} mm from {p['identity_rotation_deg']:.1f}° / "
                 f"{p['identity_translation_mm']:.1f} mm; field corr {score['field']['corr']:.3f}, rms {score['field']['rms_residual_mm']:.2f} mm",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/gre_refinement.png", dpi=130)
    plt.close(fig)


def overlay_boundary(ax, m2d, color="#ff3b1f"):
    """Draw the boundary of a binary mask slice (the gray/white boundary when it is a white-matter mask)."""
    ax.contour(m2d.astype(float), levels=[0.5], colors=[color], linewidths=0.7, origin="lower")


def best_slice(disp, mask):
    """The axial slice with the largest mean displacement among those that show most of the brain."""
    area = mask.sum(axis=(0, 1))
    cands = [(np.abs(disp[:, :, k][mask[:, :, k]]).mean(), k) for k in range(mask.shape[2]) if area[k] > 0.88 * area.max()]
    return max(cands)[1]


def fieldmapless_sbref(real, real_out, wm_path, out):
    sb, img = vol(f"{real}/sbref_ap.nii.gz")
    wm = vol(wm_path)[0] > 0.5
    corr, _ = vol(f"{real_out}/corrected.nii.gz")
    t2, _ = vol(f"{real}/t2w.nii.gz")
    field, _ = vol(f"{real_out}/field_hz.nii.gz")
    topup, _ = vol(f"{real}/se_topup_field_hz.nii.gz")
    disp, _ = vol(f"{real_out}/disp_pe_mm.nii.gz")
    mask = vol(f"{real}/brain_mask.nii.gz")[0] > 0
    z = best_slice(disp, mask)
    aspect = img.header.get_zooms()[1] / img.header.get_zooms()[0]
    m2 = axial(mask, z)
    w2 = crop(axial(wm, z), m2, 8)
    field = field - field[mask].mean()  # the fit pins the in-brain mean (the pose carries the DC); compare structure
    topup = topup - topup[mask].mean()
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.8))
    for ax, a, t in zip(axes[0], [sb, corr, t2],
                        ["SBRef as acquired\n(white-matter boundary in red)", "corrected by unpile\n(T2w reference, joint pose)", "T2w reference"]):
        panel(ax, crop(axial(a, z), m2, 8), t, aspect=aspect)
        if a is not t2:
            overlay_boundary(ax, w2)
    hz = 70
    for ax, a, t in zip(axes[1], [field, topup, field - topup],
                        ["unpile field (Hz, mean removed)", "FSL topup field from the\nspin-echo pair (Hz, mean removed)", "difference (Hz)"]):
        im = panel(ax, crop(axial(masked(a, mask), z), m2, 8), t, hz=hz, aspect=aspect)
    cb = fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.02)
    cb.set_label("Hz", fontsize=8)
    d = np.abs(disp[:, :, z][m2])
    fig.suptitle(f"Fieldmap-less correction of a 2.4 mm SBRef (OpenNeuro ds002737), fitted on a 1.8 mm grid\n"
                 f"this slice: median displacement {np.median(d):.1f} mm, maximum {d.max():.1f} mm", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{out}/fieldmapless_sbref.png", dpi=130)
    plt.close(fig)


def world_cut(img, mask, axis, index):
    """World coordinate of a slice index along `axis`, taken at the mask centroid of that slice."""
    sub = np.take(mask, index, axis=axis)
    idx = np.array(np.nonzero(sub)).mean(axis=1)
    vox = np.insert(idx, axis, index)
    return float((img.affine @ np.append(vox, 1.0))[axis])


def before_after_svg(before_path, after_path, mask_path, wm_path, disp_path, out_file, labels=("as acquired", "corrected")):
    """A nireports before/after view (the kind qsiprep's reports use): two axial and two coronal panels
    through the slices where the displacement is largest, with the white-matter boundary on both images. The alternation is set to run on its own (nireports pauses it
    until the pointer hovers, which never happens to an SVG embedded as an image)."""
    from nireports.reportlets.mosaic import plot_registration
    from nireports.reportlets.utils import compose_view

    before, after = nb.load(before_path), nb.load(after_path)
    mask_img = nb.load(mask_path)
    mask = np.asarray(mask_img.dataobj) > 0
    wm = nb.load(wm_path)
    disp = np.abs(np.asarray(nb.load(disp_path).dataobj, np.float32))
    # two axial and two coronal panels through the slices with the largest mean displacement, kept
    # at least 15 mm apart: at the page's column width each panel is still about 2 pixels per mm
    def best_cuts(axis, frac, n=2, min_sep_mm=15.0):
        area = mask.sum(axis=tuple(a for a in range(3) if a != axis))
        ranked = sorted(((np.take(disp, k, axis=axis)[np.take(mask, k, axis=axis)].mean(), k) for k in range(mask.shape[axis]) if area[k] > frac * area.max()), reverse=True)
        vox_mm = float(np.linalg.norm(before.affine[:3, axis]))
        chosen = []
        for _, k in ranked:
            if all(abs(k - c) * vox_mm >= min_sep_mm for c in chosen):
                chosen.append(k)
            if len(chosen) == n:
                break
        return [world_cut(before, mask, axis, k) for k in sorted(chosen)]
    cuts = {"z": best_cuts(2, 0.88), "y": best_cuts(1, 0.5)}
    bg = plot_registration(before, "fixed-image", estimate_brightness=True, cuts=cuts, order=("z", "y"), label=labels[0], contour=wm, compress="auto")
    fg = plot_registration(after, "moving-image", estimate_brightness=True, cuts=cuts, order=("z", "y"), label=labels[1], contour=wm, compress="auto")
    compose_view(bg, fg, out_file=out_file)
    txt = open(out_file).read().replace("alternate none infinite paused", "alternate none infinite running").replace("animation: 1s ease-in-out", "animation: 2s ease-in-out")
    open(out_file, "w").write(txt)


def csdsi2_panels(cs, t2w_out, gre_out, out):
    """Real 1.7 mm diffusion data with a 92 ms readout: the b=0 corrected without a fieldmap (T2w) and
    with the displaced fieldmap recovered against a same-contrast reference, next to the T2w; below,
    the fields next to topup's from the AP/PA pair."""
    b0, img = vol(f"{cs}/AP_b0_mean.nii.gz")
    t2, _ = vol(f"{cs}/T2w_in_b0.nii.gz")
    c_t2w, _ = vol(f"{t2w_out}/corrected.nii.gz")
    c_gre, _ = vol(f"{gre_out}/corrected.nii.gz")
    f_t2w, _ = vol(f"{t2w_out}/field_hz.nii.gz")
    f_gre, _ = vol(f"{gre_out}/field_hz.nii.gz")
    g_moved, _ = vol(f"{cs}/gre_hz_in_b0_moved5deg4mm.nii.gz")
    topup, _ = vol(f"{cs}/topup_field_hz.nii.gz")
    disp, _ = vol(f"{gre_out}/disp_pe_mm.nii.gz")
    mask = vol(f"{cs}/brain_mask.nii.gz")[0] > 0
    wm = vol(f"{cs}/wm_mask.nii.gz")[0] > 0.5
    z = best_slice(disp, mask)
    aspect = img.header.get_zooms()[1] / img.header.get_zooms()[0]
    m2 = axial(mask, z)
    w2 = crop(axial(wm, z), m2, 6)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6))
    for ax, a, t in zip(axes[0], [b0, c_t2w, c_gre, t2],
                        ["b=0 as acquired\n(white-matter boundary in red)", "corrected without a fieldmap\n(T2w reference)",
                         "corrected with the fieldmap\n(displaced 5° / 4 mm, then refined)", "T2w"]):
        panel(ax, crop(axial(a, z), m2, 6), t, aspect=aspect)
        if a is not t2:
            overlay_boundary(ax, w2)
    hz = 80
    fields = [f_t2w - f_t2w[mask].mean(), g_moved - g_moved[mask].mean(), f_gre - f_gre[mask].mean(), topup - topup[mask].mean()]
    for ax, a, t in zip(axes[1], fields, ["field without a fieldmap (Hz)", "fieldmap as displaced (Hz)", "fieldmap after refinement (Hz)", "FSL topup from the AP/PA pair (Hz)"]):
        im = panel(ax, crop(axial(masked(a, mask), z), m2, 6), t, hz=hz, aspect=aspect)
    cb = fig.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.02)
    cb.set_label("Hz, mean removed", fontsize=8)
    d = np.abs(disp[:, :, z][m2])
    fig.suptitle(f"1.7 mm diffusion b=0 with a 92 ms readout (HBCD protocol); this slice: displacement {np.median(d):.1f} mm at the median voxel, {np.percentile(d, 90):.1f} mm at the 90th percentile", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/csdsi2_panels.png", dpi=130)
    plt.close(fig)


def resolution_sweep(out):
    # from docs/user/fieldmapless.md: fields scored on the native grid in the same mask
    pd50 = [("3 (native)", 0.543, 3.92), ("2", 0.760, 2.71), ("1.5", 0.748, 2.72), ("1.2", 0.800, 2.45)]
    sbref = [("2.4 (native)", 0.656, 1.82), ("1.8", 0.740, 1.62), ("1.5", 0.713, 1.69), ("1.2", 0.711, 1.69)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, rows, title in zip(axes, [pd50, sbref], ["TRXScan phasediff fixture (3 mm EPI), T2w reference, vs true field",
                                                     "ds002737 SBRef (2.4 mm), T2w reference, vs spin-echo topup field"]):
        x = np.arange(len(rows))
        colors = ["#9db4c0" if i != 1 else "#2a6f97" for i in range(len(rows))]
        ax.bar(x, [r[1] for r in rows], color=colors)
        for i, r in enumerate(rows):
            ax.text(i, r[1] + 0.01, f"{r[1]:.2f}\n{r[2]:.2f} mm", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [r[0] for r in rows])
        ax.set_xlabel("fitting grid (mm)", fontsize=9)
        ax.set_ylim(0.4, 0.95)
        ax.set_ylabel("field correlation", fontsize=9)
        ax.set_title(title, fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fitting grid: two thirds of the EPI voxel captures the gain (dark bar); rms residual below each bar", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{out}/resolution_sweep.png", dpi=130)
    plt.close(fig)


def two_blip(fx, out):
    d = os.path.join(fx, "rpe", "unpile")
    ap, img = vol(f"{d}/b0_AP.nii.gz")
    pa, _ = vol(f"{d}/b0_PA.nii.gz")
    corr, _ = vol(f"{d}/out/pepolar/corrected.nii.gz")
    clean, _ = vol(f"{d}/clean_b0.nii.gz")
    mask = vol(f"{d}/brain_mask.nii.gz")[0] > 0
    z = centre_slice(mask)
    aspect = img.header.get_zooms()[1] / img.header.get_zooms()[0]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    m2 = axial(mask, z)
    for ax, a, t in zip(axes, [ap, pa, corr, clean], ["b=0, AP", "b=0, PA", "corrected AP", "simulated truth (no distortion)"]):
        panel(ax, crop(axial(a, z), m2, 6), t, aspect=aspect)
    fig.suptitle("Two opposite-polarity images (TRXScan rpe fixture, 3 mm)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/two_blip.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="testdata/fixtures")
    ap.add_argument("--testdata-real", required=True, help="ds002737 inputs on the 1.8 mm grid")
    ap.add_argument("--real-out", required=True, help="unpile output directory of the 1.8 mm ds002737 fit")
    ap.add_argument("--wm", required=True, help="white-matter mask on the 1.8 mm grid (SynthSeg of the T2w, labels 2/41/7/46/16)")
    ap.add_argument("--b5k", required=True, help="BOLD5000 ses-01 inputs from the test data bundle (se_ap, t2w, brain_mask)")
    ap.add_argument("--b5k-out", required=True, help="unpile output directory of the BOLD5000 ses-01 T2w fit (native grid)")
    ap.add_argument("--b5k-wm", required=True, help="white-matter mask on the BOLD5000 ses-01 grid (SynthSeg of its T2w)")
    ap.add_argument("--csdsi2", required=True, help="CSDSI2 inputs on the b=0 grid (AP_b0_mean, T2w_in_b0, brain_mask, wm_mask, gre_hz_in_b0_moved5deg4mm, topup_field_hz)")
    ap.add_argument("--csdsi2-t2w-out", required=True, help="unpile output of the T2w fieldmap-less fit")
    ap.add_argument("--csdsi2-gre-out", required=True, help="unpile output of the displaced-fieldmap refinement with a same-contrast reference")
    ap.add_argument("--out", default="docs/_static")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gre_refinement(a.fixtures, a.out)
    fieldmapless_sbref(a.testdata_real, a.real_out, a.wm, a.out)
    before_after_svg(f"{a.b5k}/se_ap.nii.gz", f"{a.b5k_out}/corrected.nii.gz", f"{a.b5k}/brain_mask.nii.gz", a.b5k_wm,
                     f"{a.b5k_out}/disp_pe_mm.nii.gz", f"{a.out}/fieldmapless_before_after.svg", ("as acquired", "corrected"))
    fx = os.path.join(a.fixtures, "phasediff", "unpile")
    before_after_svg(f"{fx}/b0_AP.nii.gz", f"{fx}/out/gre/corrected.nii.gz", f"{fx}/brain_mask.nii.gz", f"{fx}/wm_mask.nii.gz",
                     f"{fx}/out/gre/disp_pe_mm.nii.gz", f"{a.out}/gre_refinement_before_after.svg", ("as acquired", "corrected"))
    resolution_sweep(a.out)
    two_blip(a.fixtures, a.out)
    csdsi2_panels(a.csdsi2, a.csdsi2_t2w_out, a.csdsi2_gre_out, a.out)
    before_after_svg(f"{a.csdsi2}/AP_b0_mean.nii.gz", f"{a.csdsi2_t2w_out}/corrected.nii.gz", f"{a.csdsi2}/brain_mask.nii.gz", f"{a.csdsi2}/wm_mask.nii.gz",
                     f"{a.csdsi2_t2w_out}/disp_pe_mm.nii.gz", f"{a.out}/csdsi2_fieldmapless_before_after.svg", ("as acquired", "corrected"))
    before_after_svg(f"{a.csdsi2}/AP_b0_mean.nii.gz", f"{a.csdsi2_gre_out}/corrected.nii.gz", f"{a.csdsi2}/brain_mask.nii.gz", f"{a.csdsi2}/wm_mask.nii.gz",
                     f"{a.csdsi2_gre_out}/disp_pe_mm.nii.gz", f"{a.out}/csdsi2_refinement_before_after.svg", ("as acquired", "corrected"))
    print("figures written to", a.out)
