#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Figures for docs/user/how-it-works.md: the estimation followed on one phase-encoding line.

    python docs/tools/make_line_walkthrough.py --testdata testdata/rpe --out docs/_static

Needs torch, nibabel and matplotlib. The line operators live in line_model.py; before drawing
anything this script checks them against the PyTorch prototype's outputs in testdata/rpe. It
prints the numbers quoted on the page.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import torch
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(__file__))
import line_model as lm  # noqa: E402

LINE = (25, 18)  # x and z of the line: midline, inferior slice through the frontal artefact
LEVELS_MM = (24, 12, 6, 3)  # the CLI's default control-grid schedule
ITERS = 60
LAM = 0.1
SEG = (34, 62)  # the part of the line the close-ups show

REF_C, OBS_C, PRED_C, TRUE_C, FIT_C = "0.15", "#c0392b", "#2471a3", "0.15", "#e67e22"
X_LABEL = "voxel index along the phase-encoding axis (3 mm voxels, posterior → anterior)"
SIG_LABEL = "signal\n(median brain signal = 1)"
DISP_LABEL = "displacement (mm)\n+ = anterior"

plt.rcParams.update({
    "svg.fonttype": "none",  # text stays text: smaller files, searchable labels
    "svg.hashsalt": "unpile-line-walkthrough",  # stable ids, so regenerating gives a quiet diff
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def load_line(testdata):
    meta = json.load(open(os.path.join(testdata, "meta.json")))
    assert meta["pe_axis"] == 1, "the walkthrough expects the PE axis to be j"
    x, z = LINE

    def vol(name):
        return np.asarray(nb.load(os.path.join(testdata, name)).dataobj, np.float32)

    vols = {k: vol(f"{k}.nii.gz") for k in ("b0", "clean", "field_true", "brain_mask")}
    brain = torch.from_numpy(vols["brain_mask"][x, :, z] > 0)
    obs = torch.from_numpy(vols["b0"][x, :, z].copy())
    ref = torch.from_numpy(vols["clean"][x, :, z].copy())
    obs, ref = obs / obs[brain].median(), ref / ref[brain].median()
    # the data term is evaluated in the brain mask dilated by 4 voxels, as in `unpile fit`
    near = torch.from_numpy(np.convolve(brain.numpy(), np.ones(9), "same") > 0).float()
    # the field is pulled to zero more than 3 voxels outside the head (signal above 8% of the 99th centile)
    head = (obs > 0.08 * torch.quantile(obs[brain], 0.99)) | brain
    outside = torch.from_numpy(np.convolve(head.numpy(), np.ones(7), "same") == 0).float()
    return dict(
        obs=obs, ref=ref, brain=brain, near=near, outside=outside,
        field_true=torch.from_numpy(vols["field_true"][x, :, z].copy()),
        step=meta["sign"] * meta["trt"],  # displacement in voxels per Hz
        vox=float(meta["vox"][1]), trt=meta["trt"], vols=vols,
    )


def loss_terms(f, d, sigma, lam=LAM):
    """The objective `unpile fit` minimises, on one line: data + lam * bending + outside penalty."""
    pred = lm.gaussian_smooth(lm.splat(d["ref"], f * d["step"]), sigma)
    data, gain = lm.ssd_with_gain(pred, lm.gaussian_smooth(d["obs"], sigma), d["near"])
    bend = lm.bending_energy(f, d["vox"])
    fo = f / lm.SCALE
    out = (d["outside"] * fo * fo).sum() / d["outside"].sum()
    return data + lam * bend + out, data, bend, gain


def fit(d):
    """Coarse to fine, L-BFGS on B-spline coefficients; a snapshot after every iteration."""
    n = d["obs"].shape[0]
    f_full = torch.zeros(n)
    snaps = []
    for level, h_mm in enumerate(LEVELS_MM):
        M = lm.bspline_matrix(n, max(round(h_mm / d["vox"]), 1))
        c = (torch.linalg.pinv(M) @ (f_full / lm.SCALE)).requires_grad_(True)
        sigma = min(0.5 * h_mm / d["vox"], 3.0) if level < len(LEVELS_MM) - 1 else 0.0
        # max_iter=1 per call so every iteration can be recorded; the history carries over between
        # calls. max_eval must be set: its default (max_iter * 5 // 4) would allow one line-search step.
        opt = torch.optim.LBFGS([c], lr=1, max_iter=1, max_eval=25, history_size=20,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            loss = loss_terms(M @ c * lm.SCALE, d, sigma)[0]
            loss.backward()
            return loss

        for it in range(ITERS + 1):
            if it:
                opt.step(closure)
            f = (M @ c * lm.SCALE).detach()
            loss, data, bend, gain = loss_terms(f, d, sigma)
            snaps.append(dict(level=level, h_mm=h_mm, it=it, sigma=sigma, field=f, loss=loss.item(),
                              data=data.item(), bend=bend.item(), gain=gain.item(),
                              pred=gain * lm.splat(d["ref"], f * d["step"])))
        f_full = f
    return snaps


def mm(field_hz, d):
    return field_hz * d["step"] * d["vox"]


def rms(a, sel):
    return (a[sel] ** 2).mean().sqrt().item()


def disp_error(s, d):
    return rms(mm(s["field"], d) - mm(d["field_true"], d), d["brain"])


def shade_brain(ax, d, color="0.94"):
    ys = np.where(d["brain"].numpy())[0]
    ax.axvspan(ys.min() - 0.5, ys.max() + 0.5, color=color, zorder=-5, lw=0)


def fig_setup(d, out):
    """The line on the images, the two profiles and the true displacement."""
    x, z = LINE
    v = d["vols"]
    fig = plt.figure(figsize=(10, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 3.2], hspace=0.4, wspace=0.55)
    for col, (key, title) in enumerate([("clean", "A  undistorted reference\n"), ("b0", "B  acquired b=0\n")]):
        ax = fig.add_subplot(gs[:, col])
        img = v[key][:, :, z].T
        ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=np.percentile(img, 99.5))
        ax.axvline(x, color=OBS_C if key == "b0" else "#f4d03f", lw=1.2)
        ax.set_title(title)
        ax.text(0.5, 1.0, "anterior", transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="0.4")
        ax.text(0.5, 0.0, "posterior", transform=ax.transAxes, ha="center", va="top", fontsize=7, color="0.4")
        ax.set_axis_off()
    y = np.arange(d["obs"].shape[0])
    ax = fig.add_subplot(gs[0, 2])
    shade_brain(ax, d)
    ax.plot(y, d["ref"], color=REF_C, lw=1.4, label="undistorted reference")
    ax.plot(y, d["obs"], color=OBS_C, lw=1.4, label="acquired b=0")
    ax.set_ylabel(SIG_LABEL)
    ax.legend(loc="upper left")
    ax.set_title("C  signal along the line")
    ax = fig.add_subplot(gs[1, 2], sharex=ax)
    shade_brain(ax, d)
    jac = 1 + lm.ddx(d["field_true"] * d["step"])
    ax.plot(y, mm(d["field_true"], d), color=TRUE_C, lw=1.4)
    ax.axhline(0, color="0.6", lw=0.6)
    lo, hi = ax.get_ylim()
    ax.fill_between(y, lo, hi, where=(jac < 0.85).numpy(), color=OBS_C, alpha=0.15, lw=0,
                    label="compressed (local width < 85%)")
    ax.fill_between(y, lo, hi, where=(jac > 1.15).numpy(), color=PRED_C, alpha=0.15, lw=0,
                    label="stretched (local width > 115%)")
    ax.set_ylim(lo, hi)
    ax.set_ylabel(DISP_LABEL)
    ax.set_xlabel(X_LABEL)
    ax.legend(loc="lower left")
    ax.set_title("D  true displacement")
    save(fig, out, "line_setup.svg")


def fig_cells(d, out):
    """Each voxel of the reference as a cell, moved by the true displacement, then binned."""
    s0, s1 = SEG
    disp = d["field_true"] * d["step"]
    a, b = lm.moved_cells(disp)
    W = lm.splat_weights(disp)
    gain = lm.ssd_with_gain(lm.splat(d["ref"], disp), d["obs"], d["near"])[1].item()
    ref = d["ref"].numpy()
    colors = plt.cm.tab20(np.arange(len(ref)) % 20)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True, gridspec_kw=dict(hspace=0.35))
    top = 1.15 * max(d["obs"][s0:s1].max().item(), (d["ref"] / (b - a))[s0:s1].max().item())
    for i in range(len(ref)):
        if s0 <= i < s1:
            axes[0].add_patch(Rectangle((i - 0.5, 0), 1, ref[i], color=colors[i], lw=0.3, ec="white"))
        width = (b[i] - a[i]).item()
        if b[i] > s0 - 0.5 and a[i] < s1 - 0.5:  # cells from outside the window can land inside it
            axes[1].add_patch(Rectangle((a[i].item(), 0), width, ref[i] / width, color=colors[i], lw=0.3, ec="white"))
    contrib = (d["ref"][:, None] * W).numpy()  # [source cell, output voxel]
    for j in range(s0, s1):
        bottom = 0.0
        for i in np.nonzero(contrib[:, j] > 1e-3)[0]:
            axes[2].add_patch(Rectangle((j - 0.4, bottom), 0.8, contrib[i, j], color=colors[i], lw=0.3, ec="white"))
            bottom += contrib[i, j]
    axes[2].plot(np.arange(s0, s1), d["obs"][s0:s1] / gain, "o", color="black", ms=3.5,
                 label="acquired b=0, divided by the fitted intensity gain")
    axes[2].legend(loc="upper right")
    titles = ["A  reference: one cell per voxel, area = signal",
              "B  cell edges displaced; areas preserved",
              "C  prediction: area collected by each voxel, coloured by source cell"]
    for ax, t in zip(axes, titles):
        ax.set_ylim(0, top)
        ax.set_title(t)
        ax.set_ylabel("signal density")
        for j in range(s0, s1 + 1):
            ax.axvline(j - 0.5, color="0.85", lw=0.5, zorder=-1)
    axes[2].set_ylabel(SIG_LABEL)
    axes[2].set_xlim(s0 - 0.5, s1 - 0.5)
    axes[2].set_xlabel(X_LABEL)
    save(fig, out, "line_cells.svg")
    return contrib


def fig_push_pull(d, out):
    """At the true field: predicting the acquisition (push) against undoing it (pull)."""
    disp = d["field_true"] * d["step"]
    push = lm.splat(d["ref"], disp)
    gain = lm.ssd_with_gain(push, d["obs"], d["near"])[1]
    push, pull_jac = gain * push, lm.pullback(d["obs"], disp) / gain
    y = np.arange(len(push))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    ax = axes[0]
    shade_brain(ax, d)
    ax.plot(y, d["obs"], color=OBS_C, lw=1.4, label="acquired b=0")
    ax.plot(y, push, "--", color=PRED_C, lw=1.4, label="reference, pushed forward (× gain)")
    ax.set_title("A  push: reference → acquisition")
    ax.set_ylabel(SIG_LABEL)
    ax = axes[1]
    shade_brain(ax, d)
    ax.plot(y, d["ref"], color=REF_C, lw=1.4, label="undistorted reference")
    ax.plot(y, pull_jac, "--", color=FIT_C, lw=1.4, label="acquired, resampled × Jacobian (÷ gain)")
    ax.set_title("B  pull: acquisition → reference")
    for ax in axes:
        ax.legend(loc="upper left")
        ax.set_xlabel("voxel index (posterior → anterior)")
        ax.set_xlim(*SEG)
    save(fig, out, "line_push_pull.svg")
    jac = 1 + lm.ddx(disp)
    plain = lm.pull(d["obs"], disp) / gain
    nums = {}
    for name, sel in [("pileup", (jac < 0.85) & d["brain"]), ("stretched", (jac > 1.15) & d["brain"])]:
        nums[name] = dict(n=int(sel.sum()), push=rms(push - d["obs"], sel), pull_jac=rms(pull_jac - d["ref"], sel),
                          pull_plain=rms(plain - d["ref"], sel))
    return nums


def fig_gradient(d, out):
    """The first gradient: which way the fit moves the field, starting from zero."""
    f = torch.zeros_like(d["obs"]).requires_grad_(True)
    pred = lm.splat(d["ref"], f * d["step"])
    data, gain = lm.ssd_with_gain(pred, d["obs"], d["near"])
    data.backward()
    y = np.arange(len(pred))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw=dict(hspace=0.3))
    ax = axes[0]
    shade_brain(ax, d)
    ax.plot(y, d["obs"], color=OBS_C, lw=1.4, label="acquired b=0")
    ax.plot(y, (gain * pred).detach(), "--", color=PRED_C, lw=1.4, label="prediction at zero field (× gain)")
    ax.legend(loc="upper left")
    ax.set_ylabel(SIG_LABEL)
    ax.set_title("A  prediction at the starting point")
    ax = axes[1]
    shade_brain(ax, d)
    # descent direction of the field, expressed as a displacement (sign of the PE step applied)
    step = -f.grad * np.sign(d["step"])
    M = lm.bspline_matrix(len(y), round(LEVELS_MM[0] / d["vox"]))
    coarse = M @ (M.T @ step)  # the field change a gradient step on the 24 mm grid's coefficients makes
    true = mm(d["field_true"], d)
    ax.plot(y, step / step.abs().max(), color=FIT_C, lw=1.4, label="per-voxel descent direction")
    ax.plot(y, coarse / coarse.abs().max(), color=PRED_C, lw=1.4,
            label=f"the same, projected on the {LEVELS_MM[0]} mm B-spline grid")
    ax.plot(y, true / true.abs().max(), color=TRUE_C, lw=1.0, alpha=0.6, label="true displacement")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_yticks([0])
    ax.set_ylabel("direction (+ = anterior)\neach curve scaled to unit peak")
    ax.legend(loc="lower left")
    ax.set_title("B  descent direction of the objective with respect to the field")
    ax.set_xlabel(X_LABEL)
    save(fig, out, "line_gradient.svg")


def fig_steps(d, snaps, out):
    """Snapshots of the fit: prediction against the acquisition, field against the truth."""
    picks = [(0, 0), (0, 1), (0, 3), (0, 10), (0, ITERS), (len(LEVELS_MM) - 1, ITERS)]
    y = np.arange(len(d["obs"]))
    true_mm = mm(d["field_true"], d)
    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 2.1 * len(picks)), sharex=True,
                             gridspec_kw=dict(hspace=0.5, wspace=0.2))
    for row, (level, it) in enumerate(picks):
        s = next(s for s in snaps if s["level"] == level and s["it"] == it)
        ax = axes[row, 0]
        shade_brain(ax, d)
        ax.plot(y, d["obs"], color=OBS_C, lw=1.2, label="acquired b=0")
        ax.plot(y, s["pred"], "--", color=PRED_C, lw=1.2, label="prediction (× gain)")
        ax.set_title(f"{s['h_mm']} mm grid, iteration {it}")
        ax.set_ylabel("signal")
        ax = axes[row, 1]
        shade_brain(ax, d)
        ax.plot(y, true_mm, color=TRUE_C, lw=1.2, label="true")
        ax.plot(y, mm(s["field"], d), color=FIT_C, lw=1.4, label="estimated")
        ax.set_ylim(true_mm.min() - 2, true_mm.max() + 2)
        ax.set_ylabel("mm")
        ax.set_title(f"displacement; RMS error in the brain {disp_error(s, d):.1f} mm")
    axes[0, 0].legend(loc="upper left")
    axes[0, 1].legend(loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("voxel index (posterior → anterior)")
    save(fig, out, "line_fit_steps.svg")


def fig_levels(d, snaps, out):
    """The objective over the whole schedule, and the field at the end of each level."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(wspace=0.25))
    ax = axes[0]
    ax.semilogy([s["data"] for s in snaps], color=PRED_C, lw=1.3, label="data term")
    ax.semilogy([LAM * s["bend"] for s in snaps], color=FIT_C, lw=1.3, label="λ × bending energy")
    for level, h in enumerate(LEVELS_MM):
        k = next(i for i, s in enumerate(snaps) if s["level"] == level)
        ax.axvline(k, color="0.7", lw=0.6)
        ax.text(k + 1, 0.98, f"{h} mm", transform=ax.get_xaxis_transform(), fontsize=8, va="top")
    ax.set_xlabel("iteration, all levels")
    ax.set_ylabel("contribution to the objective")
    ax.legend(loc="lower left")
    ax.set_title("A  objective terms")
    ax = axes[1]
    shade_brain(ax, d)
    y = np.arange(len(d["obs"]))
    ax.plot(y, mm(d["field_true"], d), color=TRUE_C, lw=2.0, label="true")
    for level, h in enumerate(LEVELS_MM):
        s = [s for s in snaps if s["level"] == level][-1]
        ax.plot(y, mm(s["field"], d), lw=1.1, label=f"after the {h} mm grid", color=plt.cm.viridis(level / 3.5))
    ax.set_ylabel(DISP_LABEL)
    ax.set_xlabel("voxel index (posterior → anterior)")
    ax.legend(loc="lower left")
    ax.set_title("B  displacement at the end of each level")
    save(fig, out, "line_levels.svg")


def fig_unwarp(d, snaps, out):
    """The corrected line from the estimated field: inverting the push against resampling."""
    disp = snaps[-1]["field"] * d["step"]
    g = snaps[-1]["gain"]  # put the acquired intensities back on the reference's scale
    inv = lm.unwarp(d["obs"], disp) / g
    pb = lm.pullback(d["obs"], disp) / g
    plain = lm.pull(d["obs"], disp) / g
    y = np.arange(len(inv))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    shade_brain(ax, d)
    ax.plot(y, d["ref"], color=REF_C, lw=2.0, label="undistorted reference")
    ax.plot(y, plain, ":", color="0.5", lw=1.3, label="resampled (as with sdc_warp.nii.gz)")
    ax.plot(y, pb, "--", color=FIT_C, lw=1.3, label="resampled × Jacobian")
    ax.plot(y, inv, color=PRED_C, lw=1.4, label="least-squares inverse of the push (corrected.nii.gz)")
    ax.set_xlim(*SEG)
    ax.set_ylabel(SIG_LABEL)
    ax.legend(loc="upper left")
    ax.set_xlabel("voxel index (posterior → anterior)")
    save(fig, out, "line_unwarp.svg")
    pile = ((1 + lm.ddx(d["field_true"] * d["step"])) < 0.85) & d["brain"]
    return dict(plain=rms(plain - d["ref"], pile), pull_jac=rms(pb - d["ref"], pile), inverse=rms(inv - d["ref"], pile))


def restoration_limits(d):
    """How much of the pile-up one acquisition can restore, and what the opposite direction adds."""
    disp = d["field_true"] * d["step"]
    pile = ((1 + lm.ddx(disp)) < 0.85) & d["brain"]
    gain = lm.ssd_with_gain(lm.splat(d["ref"], disp), d["obs"], d["near"])[1]
    out = {}
    w_one = lm.splat_weights(disp)
    w_two = torch.cat([w_one, lm.splat_weights(-disp)], 1)
    for name, w in [("one_direction", w_one), ("two_directions", w_two)]:
        sv = torch.linalg.svdvals(w)
        out[f"{name}_zero_singular_values"] = int((sv < 1e-3).sum())
        out[f"{name}_smallest_singular_value"] = sv.min().item()
    # noise-free: the reference pushed forward with the true field, restored without any penalty
    one = lm.restore([lm.splat(d["ref"], disp)], [disp], ridge=1e-6, grad_reg=0.0)
    two = lm.restore([lm.splat(d["ref"], disp), lm.splat(d["ref"], -disp)], [disp, -disp], ridge=1e-6, grad_reg=0.0)
    out["noise_free_one_direction"] = rms(one - d["ref"], pile)
    out["noise_free_two_directions"] = rms(two - d["ref"], pile)
    # the acquired line at the true field: the penalty, not the model, makes the difference
    for gr in (0.05, 0.0):
        out[f"acquired_inverse_grad_reg_{gr}"] = rms(lm.unwarp(d["obs"], disp, grad_reg=gr) / gain - d["ref"], pile)
    out["acquired_pull_jacobian"] = rms(lm.pullback(d["obs"], disp) / gain - d["ref"], pile)
    return out


def animation(d, snaps, out, dt=0.12, hold=2.0):
    """The fit as an animated SVG: every frame is drawn once and shown in turn by CSS keyframes."""
    keep = [s for s in snaps if s["it"] <= 12 or s["it"] % 6 == 0]
    y = np.arange(len(d["obs"]))
    true_mm = mm(d["field_true"], d)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True, gridspec_kw=dict(hspace=0.35))
    for ax in axes:
        shade_brain(ax, d)
    axes[0].plot(y, d["obs"], color=OBS_C, lw=1.3, label="acquired b=0")
    axes[0].plot([], [], "--", color=PRED_C, lw=1.3, label="prediction (× gain)")
    axes[0].legend(loc="upper left")
    axes[0].set_ylabel("signal")
    axes[1].plot(y, true_mm, color=TRUE_C, lw=1.3, label="true")
    axes[1].plot([], [], color=FIT_C, lw=1.5, label="estimated")
    axes[1].set_ylim(true_mm.min() - 2, true_mm.max() + 2)
    axes[1].set_ylabel(DISP_LABEL)
    axes[1].set_xlabel("voxel index (posterior → anterior)")
    axes[1].legend(loc="lower left")
    axes[0].set_ylim(axes[0].get_ylim())
    for k, s in enumerate(keep):
        (p,) = axes[0].plot(y, s["pred"], "--", color=PRED_C, lw=1.3)
        (f,) = axes[1].plot(y, mm(s["field"], d), color=FIT_C, lw=1.5)
        t = axes[0].text(0, 1.04, f"{s['h_mm']} mm grid, iteration {s['it']}", transform=axes[0].transAxes)
        e = axes[1].text(0, 1.04, f"RMS displacement error in the brain {disp_error(s, d):.1f} mm",
                         transform=axes[1].transAxes)
        for part, artist in zip("pfte", (p, f, t, e)):
            artist.set_gid(f"frame{k}{part}")
    path = os.path.join(out, "line_fit.svg")
    save(fig, out, "line_fit.svg")
    n = len(keep)
    total = n * dt + hold
    css = ["<style type=\"text/css\">", "[id^=\"frame\"] { opacity: 0; }"]
    for k in range(n):
        start = 100 * k * dt / total
        end = 100 * ((k + 1) * dt if k < n - 1 else total) / total
        css.append(f"@keyframes f{k} {{ 0%, {max(start - 0.001, 0):.3f}% {{ opacity: 0; }} "
                   f"{start:.3f}%, {end - 0.001:.3f}% {{ opacity: 1; }} {end:.3f}%, 100% {{ opacity: 0; }} }}")
        sel = ", ".join(f"#frame{k}{part}" for part in "pfte")
        css.append(f"{sel} {{ animation: f{k} {total:.2f}s linear infinite; }}")
    # without animation support the last frame is the one shown
    css.append(", ".join(f"#frame{n - 1}{part}" for part in "pfte") + " { opacity: 1; }")
    css.append("</style>")
    svg = open(path).read()
    head_end = svg.index(">", svg.index("<svg")) + 1
    open(path, "w").write(svg[:head_end] + "\n" + "\n".join(css) + svg[head_end:])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testdata", default="testdata/rpe")
    p.add_argument("--out", default="docs/_static")
    args = p.parse_args()
    diffs = lm.check(args.testdata)
    print("line operators against the prototype outputs, max abs difference (median brain signal = 1):", diffs)
    assert max(diffs.values()) < 1e-4, "the PyTorch line model no longer matches the prototype"
    d = load_line(args.testdata)
    snaps = fit(d)
    fig_setup(d, args.out)
    contrib = fig_cells(d, args.out)
    pp = fig_push_pull(d, args.out)
    fig_gradient(d, args.out)
    fig_steps(d, snaps, args.out)
    fig_levels(d, snaps, args.out)
    uw = fig_unwarp(d, snaps, args.out)
    animation(d, snaps, args.out)

    true_mm = mm(d["field_true"], d)
    jac = 1 + lm.ddx(d["field_true"] * d["step"])
    b = d["brain"]
    print(f"1 Hz = {abs(d['step']):.4f} voxel = {abs(d['step']) * d['vox']:.3f} mm; peak field "
          f"{d['field_true'][b].max():.0f} Hz; displacement {true_mm[b].min():.1f} to {true_mm[b].max():.1f} mm")
    print(f"smallest local width {jac[b].min():.2f} at voxel {int(torch.nonzero(b)[0]) + int(jac[b].argmin())}; "
          f"acquired peak {d['obs'].max():.2f} at voxel {int(d['obs'].argmax())}; reference max 38-52 {d['ref'][38:52].max():.2f}")
    top = int(d["obs"].argmax())
    print("cells landing in the peak voxel (>5% of it):",
          [i for i in range(len(d["obs"])) if contrib[i, top] > 0.05 * contrib[:, top].sum()])
    print("RMS at the true field (median brain signal = 1):", pp)
    print("corrected line vs reference in pile-up voxels:", uw)
    print("restoration limits (true field, pile-up voxels):", restoration_limits(d))
    for level, h in enumerate(LEVELS_MM):
        for s in snaps:
            if s["level"] == level and (s["it"] in (0, 1, 3, 10, ITERS) if level == 0 else s["it"] == ITERS):
                print(f"{h:>2} mm grid, iteration {s['it']:>2}: smoothing {s['sigma']:.0f} voxels, "
                      f"RMS displacement error {disp_error(s, d):.2f} mm, gain {s['gain']:.3f}")


if __name__ == "__main__":
    main()
