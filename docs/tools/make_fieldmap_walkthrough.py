#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Figures for the line walkthrough on docs/user/gre-refinement.md: refining a degraded fieldmap.

    python docs/tools/make_fieldmap_walkthrough.py --testdata testdata/rpe --out docs/_static

The same line as docs/user/how-it-works.md. A fieldmap is made from the true field by lowering its
resolution, moving it, changing its shim and adding noise inside the head; it is then refined by
the one-dimensional version of unpile's fieldmap prior (src/problem.rs, prior_field): the fieldmap
moved along the line, plus a shim polynomial, plus a smooth residual. Prints the numbers quoted on
the page.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import line_model as lm  # noqa: E402
import make_line_walkthrough as w  # noqa: E402

SHIFT_MM = 4.0  # the fieldmap is placed this far anterior of where it belongs
SHIM_HZ = (12.0, 40.0, 0.0)  # added shim: constant, linear (Hz per unit of u), quadratic
NOISE_HZ = 3.0  # standard deviation inside the head; the fieldmap is masked outside it
BLOCK = 3  # voxels averaged into one fieldmap sample: 9 mm along the line
RIDGE = 1.0  # --fieldmap-ridge default
TERMS = {"as given": (), "pose + shim": ("pose", "shim"), "pose + shim + residual": ("pose", "shim", "delta")}


def scale_mm(testdata, vox):
    import nibabel as nb

    shape = nb.load(os.path.join(testdata, "b0.nii.gz")).shape[:3]
    return max((s - 1) / 2 * vox for s in shape)


def degrade(d, basis):
    """The fieldmap's four errors, each on its own and all together (Hz)."""
    ft = d["field_true"]
    n = len(ft)
    y = np.arange(n)
    k = n // BLOCK * BLOCK
    samples = ft[:k].reshape(-1, BLOCK).mean(1).numpy()
    centres = np.arange(len(samples)) * BLOCK + (BLOCK - 1) / 2
    lowres = torch.from_numpy(np.interp(y, centres, samples)).float()
    moved = lm.shift_line(ft, SHIFT_MM / d["vox"])
    shim = 100 * (torch.tensor(SHIM_HZ) / 100) @ basis
    head = d["outside"] == 0
    noise = torch.randn(n, generator=torch.Generator().manual_seed(1)) * NOISE_HZ * head
    outside_noise = torch.randn(n, generator=torch.Generator().manual_seed(2)) * 15.0 * ~head
    combined = lm.shift_line(lowres, SHIFT_MM / d["vox"]) + shim + noise
    return {
        f"{BLOCK * d['vox']:.0f} mm resolution": lowres,
        f"moved {SHIFT_MM:.0f} mm": moved,
        "different shim": ft + shim,
        f"noise ({NOISE_HZ:.0f} Hz, inside the head)": ft + noise,
        "unmasked: 15 Hz noise outside the head only": ft + outside_noise,
        "all four": combined,
    }


def refine(d, gre, basis, terms, lam=w.LAM, iters=w.ITERS):
    """Fit pose (a shift along the line), shim and residual to the acquisition; returns the parts (Hz)."""
    n = len(gre)
    pose = torch.zeros(1, requires_grad=True)  # units of 2 mm, as pose.rs POSE_TRANS
    shim = torch.zeros(3, requires_grad=True)  # units of 100 Hz, as the prior's shim
    resid = torch.zeros(n)
    for level, h_mm in enumerate(w.LEVELS_MM):
        if not terms:
            break
        M = lm.bspline_matrix(n, max(round(h_mm / d["vox"]), 1))
        c = (torch.linalg.pinv(M) @ (resid / lm.SCALE)).requires_grad_(True)
        sigma = min(0.5 * h_mm / d["vox"], 3.0) if level < len(w.LEVELS_MM) - 1 else 0.0
        params = [p for p, k in ((pose, "pose"), (shim, "shim"), (c, "delta")) if k in terms]
        opt = torch.optim.LBFGS(params, lr=1, max_iter=1, max_eval=25, history_size=20,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

        def parts():
            r = M @ c * lm.SCALE if "delta" in terms else torch.zeros(n)
            return lm.shift_line(gre, 2 * pose[0] / d["vox"]), lm.SCALE * shim @ basis, r

        def closure():
            opt.zero_grad()
            posed, sh, r = parts()
            f = posed + sh + r
            pred = lm.gaussian_smooth(lm.splat(d["ref"], f * d["step"]), sigma)
            data, _ = lm.ssd_with_gain(pred, lm.gaussian_smooth(d["obs"], sigma), d["near"])
            fo = f / lm.SCALE
            loss = (data + lam * lm.bending_energy(r, d["vox"]) + RIDGE * ((r / lm.SCALE) ** 2).mean()
                    + (d["outside"] * fo * fo).sum() / d["outside"].sum())
            loss.backward()
            return loss

        for _ in range(iters):
            opt.step(closure)
        with torch.no_grad():
            resid = parts()[2]
    with torch.no_grad():
        posed = lm.shift_line(gre, 2 * pose[0] / d["vox"])
        sh = lm.SCALE * shim @ basis
    f = posed + sh + resid
    pred = lm.splat(d["ref"], f * d["step"])
    pred = pred * lm.ssd_with_gain(pred, d["obs"], d["near"])[1]
    return dict(field=f, posed=posed, shim=sh, resid=resid, shift_mm=2 * pose.item(),
                shim_coef=(lm.SCALE * shim).detach(), pred=pred)


def err(f, d):
    return w.rms(w.mm(f, d) - w.mm(d["field_true"], d), d["brain"])


def fig_input(d, gre, out):
    y = np.arange(len(gre))
    pred = lm.splat(d["ref"], gre * d["step"])
    pred = pred * lm.ssd_with_gain(pred, d["obs"], d["near"])[1]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw=dict(hspace=0.35))
    ax = axes[0]
    w.shade_brain(ax, d)
    ax.plot(y, d["field_true"], color=w.TRUE_C, lw=1.4, label="true field")
    ax.plot(y, gre, color=w.FIT_C, lw=1.4, label="fieldmap as given")
    ax.set_ylabel("field (Hz)")
    ax.legend(loc="upper left")
    ax.set_title("A  fieldmap and true field")
    ax = axes[1]
    w.shade_brain(ax, d)
    ax.plot(y, d["obs"], color=w.OBS_C, lw=1.4, label="acquired b=0")
    ax.plot(y, pred, "--", color=w.PRED_C, lw=1.4, label="reference pushed forward with the fieldmap (× gain)")
    ax.set_ylabel(w.SIG_LABEL)
    ax.legend(loc="upper left")
    ax.set_title("B  what the fieldmap predicts")
    ax.set_xlabel(w.X_LABEL)
    w.save(fig, out, "fieldmap_line_input.svg")


def fig_refine(d, fits, out):
    y = np.arange(len(d["obs"]))
    true_mm = w.mm(d["field_true"], d)
    used = [w.mm(fit["field"], d) for fit in fits.values()] + [true_mm]
    lo, hi = min(float(u.min()) for u in used) - 2, max(float(u.max()) for u in used) + 2
    fig, axes = plt.subplots(len(fits), 2, figsize=(11, 2.3 * len(fits)), sharex=True,
                             gridspec_kw=dict(hspace=0.5, wspace=0.2))
    for row, (name, fit) in enumerate(fits.items()):
        ax = axes[row, 0]
        w.shade_brain(ax, d)
        ax.plot(y, true_mm, color=w.TRUE_C, lw=1.2, label="true")
        ax.plot(y, w.mm(fit["field"], d), color=w.FIT_C, lw=1.4, label="field used")
        ax.set_ylim(lo, hi)
        ax.set_ylabel("mm")
        ax.set_title(f"{name}: displacement, RMS error {err(fit['field'], d):.2f} mm")
        ax = axes[row, 1]
        w.shade_brain(ax, d)
        ax.plot(y, d["obs"], color=w.OBS_C, lw=1.2, label="acquired b=0")
        ax.plot(y, fit["pred"], "--", color=w.PRED_C, lw=1.2, label="prediction (× gain)")
        ax.set_ylabel("signal")
        ax.set_title("prediction against the acquired line")
    axes[0, 0].legend(loc="lower left")
    axes[0, 1].legend(loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("voxel index (posterior → anterior)")
    w.save(fig, out, "fieldmap_line_refine.svg")


def fig_terms(d, gre, fit, out):
    """What each term changed, against what needed changing (all in Hz)."""
    y = np.arange(len(gre))
    needed = d["field_true"] - gre
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w.shade_brain(ax, d)
    ax.plot(y, needed, color=w.TRUE_C, lw=2.0, label="needed: true field − fieldmap")
    ax.plot(y, fit["posed"] - gre, color=plt.cm.viridis(0.15), lw=1.3, label=f"pose: fieldmap moved {fit['shift_mm']:+.1f} mm")
    ax.plot(y, fit["shim"], color=plt.cm.viridis(0.5), lw=1.3, label="shim polynomial")
    ax.plot(y, fit["resid"], color=plt.cm.viridis(0.85), lw=1.3, label="residual")
    ax.plot(y, fit["field"] - gre, "--", color=w.FIT_C, lw=1.4, label="sum of the three")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("change to the fieldmap (Hz)")
    ax.set_xlabel(w.X_LABEL)
    ax.legend(loc="lower left", ncol=2)
    w.save(fig, out, "fieldmap_line_terms.svg")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testdata", default="testdata/rpe")
    p.add_argument("--out", default="docs/_static")
    args = p.parse_args()
    d = w.load_line(args.testdata)
    basis = lm.shim_basis(len(d["obs"]), d["vox"], scale_mm(args.testdata, d["vox"]))
    fieldmaps = degrade(d, basis)
    gre = fieldmaps["all four"]
    fits = {name: refine(d, gre, basis, terms) for name, terms in TERMS.items()}
    fig_input(d, gre, args.out)
    fig_refine(d, fits, args.out)
    fig_terms(d, gre, fits["pose + shim + residual"], args.out)
    b = d["brain"]
    print(f"fieldmap: {BLOCK * d['vox']:.0f} mm samples, moved {SHIFT_MM} mm, shim {SHIM_HZ} Hz, noise {NOISE_HZ} Hz")
    for name, fit in fits.items():
        corr = np.corrcoef(fit["field"][b], d["field_true"][b])[0, 1]
        print(f"{name:24s} RMS {err(fit['field'], d):.2f} mm, corr {corr:.3f}, shift {fit['shift_mm']:+.2f} mm, "
              f"shim {fit['shim_coef'].numpy().round(1)} Hz")
    print("one error at a time: as given / pose + shim / pose + shim + residual (RMS mm; shift mm)")
    for name, g in fieldmaps.items():
        row = [f"{err(g, d):.2f}"]
        for terms in (TERMS["pose + shim"], TERMS["pose + shim + residual"]):
            fit = refine(d, g, basis, terms)
            row.append(f"{err(fit['field'], d):.2f} ({fit['shift_mm']:+.1f})")
        print(f"  {name:32s}", " | ".join(row))


if __name__ == "__main__":
    main()
