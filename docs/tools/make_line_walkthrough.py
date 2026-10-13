#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Figures for docs/user/how-it-works.md: unpile's fit followed on one phase-encode line.

    python docs/tools/make_line_walkthrough.py --testdata testdata/rpe --out docs/_static

Needs torch, nibabel and matplotlib. The line operators live in line_model.py; before drawing
anything this script checks them against the PyTorch prototype's dumps in testdata/rpe.
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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(__file__))
import line_model as lm  # noqa: E402

LINE = (25, 18)  # x and z of the line: midline, through the frontal hot spot
LEVELS_MM = (24, 12, 6, 3)  # the CLI's default control-grid schedule
ITERS = 60
LAM = 0.1
SEG = (34, 62)  # the stretch of the line the close-ups show

REF_C, OBS_C, PRED_C, TRUE_C, FIT_C = "0.15", "#c0392b", "#2471a3", "0.15", "#e67e22"


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
        vox=float(meta["vox"][1]), vols=vols,
    )


def loss_terms(f, d, M, sigma, lam=LAM):
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
        # max_iter=1 per call so every iteration can be recorded; the history carries over between calls
        opt = torch.optim.LBFGS([c], lr=1, max_iter=1, max_eval=25, history_size=20,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            loss = loss_terms(M @ c * lm.SCALE, d, M, sigma)[0]
            loss.backward()
            return loss

        for it in range(ITERS + 1):
            if it:
                opt.step(closure)
            f = (M @ c * lm.SCALE).detach()
            loss, data, bend, gain = loss_terms(f, d, M, sigma)
            snaps.append(dict(level=level, h_mm=h_mm, it=it, sigma=sigma, field=f, loss=loss.item(),
                              data=data.item(), bend=bend.item(), gain=gain.item(),
                              pred=gain * lm.splat(d["ref"], f * d["step"])))
        f_full = f
    return snaps


def mm(field_hz, d):
    return field_hz * d["step"] * d["vox"]


def shade_brain(ax, d, color="0.94"):
    b = d["brain"].numpy()
    ys = np.where(b)[0]
    ax.axvspan(ys.min() - 0.5, ys.max() + 0.5, color=color, zorder=-5, lw=0)


def fig_setup(d, out):
    """The line on the images, the three profiles and the true displacement."""
    x, z = LINE
    v = d["vols"]
    fig = plt.figure(figsize=(10, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 3.2], hspace=0.35, wspace=0.25)
    for col, (key, title) in enumerate([("clean", "undistorted reference"), ("b0", "acquired b=0")]):
        ax = fig.add_subplot(gs[:, col])
        img = v[key][:, :, z].T
        ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=np.percentile(img, 99.5))
        ax.axvline(x, color=OBS_C if key == "b0" else "#f4d03f", lw=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    y = np.arange(d["obs"].shape[0])
    ax = fig.add_subplot(gs[0, 2])
    shade_brain(ax, d)
    ax.plot(y, d["ref"], color=REF_C, lw=1.4, label="undistorted reference")
    ax.plot(y, d["obs"], color=OBS_C, lw=1.4, label="acquired b=0")
    ax.set_ylabel("signal (brain median = 1)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("signal along the line, posterior → anterior", fontsize=9)
    ax = fig.add_subplot(gs[1, 2], sharex=ax)
    shade_brain(ax, d)
    disp = mm(d["field_true"], d)
    jac = 1 + lm.ddx(d["field_true"] * d["step"])
    ax.plot(y, disp, color=TRUE_C, lw=1.4)
    ax.axhline(0, color="0.6", lw=0.6)
    lo, hi = ax.get_ylim()
    ax.fill_between(y, lo, hi, where=(jac < 0.85).numpy(), color=OBS_C, alpha=0.15, lw=0, label="compressed (pile-up)")
    ax.fill_between(y, lo, hi, where=(jac > 1.15).numpy(), color=PRED_C, alpha=0.15, lw=0, label="stretched")
    ax.set_ylim(lo, hi)
    ax.set_ylabel("true displacement (mm)")
    ax.set_xlabel("voxel along the phase-encode axis")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.savefig(os.path.join(out, "line_setup.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_cells(d, out):
    """Each voxel of the reference as a cell, moved by the true displacement, then binned."""
    s0, s1 = SEG
    disp = d["field_true"] * d["step"]
    a, b = lm.moved_cells(disp)
    W = lm.splat_weights(disp)
    ref = d["ref"].numpy()
    colors = plt.cm.tab20(np.arange(len(ref)) % 20)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True, gridspec_kw=dict(hspace=0.3))
    top = 1.15 * max(d["obs"][s0:s1].max().item(), (d["ref"] / (b - a))[s0:s1].max().item())
    gain = lm.ssd_with_gain(lm.splat(d["ref"], disp), d["obs"], d["near"])[1].item()
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
                 label="acquired b=0 (÷ fitted intensity gain)")
    axes[2].legend(fontsize=8, frameon=False, loc="upper right")
    titles = [
        "1. the undistorted reference: one cell per voxel, its signal is the area",
        "2. every cell edge moves by the displacement; the area stays, so a squeezed cell gets taller",
        "3. each voxel of the prediction collects the area that landed inside it",
    ]
    for ax, t in zip(axes, titles):
        ax.set_ylim(0, top)
        ax.set_title(t, fontsize=9, loc="left")
        for j in range(s0, s1 + 1):
            ax.axvline(j - 0.5, color="0.85", lw=0.5, zorder=-1)
    axes[2].set_xlim(s0 - 0.5, s1 - 0.5)
    axes[2].set_xlabel("voxel along the phase-encode axis (posterior → anterior)")
    fig.savefig(os.path.join(out, "line_cells.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_push_pull(d, out):
    """At the true field: predicting the acquisition (push) against undoing it (pull)."""
    disp = d["field_true"] * d["step"]
    push = lm.splat(d["ref"], disp)
    gain = lm.ssd_with_gain(push, d["obs"], d["near"])[1]
    push, pull_jac = gain * push, lm.pullback(d["obs"], disp) / gain
    y = np.arange(len(push))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    ax = axes[0]
    shade_brain(ax, d)
    ax.plot(y, d["obs"], color=OBS_C, lw=1.4, label="acquired b=0")
    ax.plot(y, push, "--", color=PRED_C, lw=1.4, label="reference pushed forward (× gain)")
    ax.set_title("push: predict the acquired image from the reference", fontsize=9)
    ax = axes[1]
    shade_brain(ax, d)
    ax.plot(y, d["ref"], color=REF_C, lw=1.4, label="undistorted reference")
    ax.plot(y, pull_jac, "--", color=FIT_C, lw=1.4, label="acquired, pulled back × Jacobian (÷ gain)")
    ax.set_title("pull: undo the distortion in the acquired image", fontsize=9)
    for ax in axes:
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.set_xlabel("voxel along the phase-encode axis")
        ax.set_xlim(*SEG)
    fig.savefig(os.path.join(out, "line_push_pull.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    jac = 1 + lm.ddx(disp)
    pile = (jac < 0.85) & d["brain"]
    return dict(push_rms_pileup=((push - d["obs"])[pile] ** 2).mean().sqrt().item(),
                pull_rms_pileup=((pull_jac - d["ref"])[pile] ** 2).mean().sqrt().item(),
                plain_pull_rms_pileup=((lm.pull(d["obs"], disp) / gain - d["ref"])[pile] ** 2).mean().sqrt().item())


def fig_gradient(d, out):
    """The first gradient: which way the fit wants to move the field, starting from zero."""
    f = torch.zeros_like(d["obs"]).requires_grad_(True)
    pred = lm.splat(d["ref"], f * d["step"])
    data, gain = lm.ssd_with_gain(pred, d["obs"], d["near"])
    data.backward()
    y = np.arange(len(pred))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw=dict(hspace=0.25))
    ax = axes[0]
    shade_brain(ax, d)
    ax.plot(y, d["obs"], color=OBS_C, lw=1.4, label="acquired b=0")
    ax.plot(y, (gain * pred).detach(), "--", color=PRED_C, lw=1.4, label="prediction with no field")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("no field yet: the prediction is the reference itself", fontsize=9)
    ax = axes[1]
    shade_brain(ax, d)
    step = -f.grad  # the direction that lowers the loss
    ax.plot(y, mm(step / step.abs().max(), d) / abs(d["step"] * d["vox"]), color=FIT_C, lw=1.4,
            label="downhill direction for the field (scaled)")
    M = lm.bspline_matrix(len(y), round(LEVELS_MM[0] / d["vox"]))
    coarse = M @ (M.T @ step)  # the field change a gradient step on the 24 mm grid's coefficients makes
    ax.plot(y, coarse / coarse.abs().max() * np.sign(d["step"]), color=PRED_C, lw=1.4,
            label=f"the same step through the {LEVELS_MM[0]} mm control grid (scaled)")
    ax.plot(y, mm(d["field_true"], d) / mm(d["field_true"], d).abs().max(), color=TRUE_C, lw=1.0, alpha=0.6,
            label="true displacement (scaled)")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("negative gradient of the data term, as a displacement", fontsize=9)
    ax.set_xlabel("voxel along the phase-encode axis")
    fig.savefig(os.path.join(out, "line_gradient.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_steps(d, snaps, out):
    """Snapshots of the fit: prediction against the acquisition, field against the truth."""
    picks = [(0, 0), (0, 1), (0, 3), (0, 10), (0, ITERS), (len(LEVELS_MM) - 1, ITERS)]
    y = np.arange(len(d["obs"]))
    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 2.0 * len(picks)), sharex=True,
                             gridspec_kw=dict(hspace=0.45, wspace=0.18))
    true_mm = mm(d["field_true"], d)
    for row, (level, it) in enumerate(picks):
        s = next(s for s in snaps if s["level"] == level and s["it"] == it)
        ax = axes[row, 0]
        shade_brain(ax, d)
        ax.plot(y, d["obs"], color=OBS_C, lw=1.2, label="acquired b=0")
        ax.plot(y, s["pred"], "--", color=PRED_C, lw=1.2, label="prediction (× gain)")
        ax.set_title(f"{s['h_mm']} mm grid, iteration {it}: data term {s['data']:.4f}", fontsize=8.5, loc="left")
        ax = axes[row, 1]
        shade_brain(ax, d)
        ax.plot(y, true_mm, color=TRUE_C, lw=1.2, label="true displacement (mm)")
        ax.plot(y, mm(s["field"], d), color=FIT_C, lw=1.4, label="fitted")
        ax.set_ylim(true_mm.min() - 2, true_mm.max() + 2)
        err = ((mm(s["field"], d) - true_mm)[d["brain"]] ** 2).mean().sqrt().item()
        ax.set_title(f"rms error in the brain {err:.2f} mm", fontsize=8.5, loc="left")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    axes[0, 1].legend(fontsize=8, frameon=False, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("voxel along the phase-encode axis")
    fig.savefig(os.path.join(out, "line_fit_steps.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_levels(d, snaps, out):
    """The loss over the whole schedule, and the field at the end of each level."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw=dict(wspace=0.22))
    ax = axes[0]
    ax.semilogy([s["data"] for s in snaps], color=PRED_C, lw=1.3, label="data term")
    ax.semilogy([LAM * s["bend"] for s in snaps], color=FIT_C, lw=1.3, label="λ × bending energy")
    for level, h in enumerate(LEVELS_MM):
        k = next(i for i, s in enumerate(snaps) if s["level"] == level)
        ax.axvline(k, color="0.7", lw=0.6)
        ax.text(k + 1, 0.98, f"{h} mm", transform=ax.get_xaxis_transform(), fontsize=8, va="top")
    ax.set_xlabel("iteration (all levels)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax = axes[1]
    shade_brain(ax, d)
    y = np.arange(len(d["obs"]))
    ax.plot(y, mm(d["field_true"], d), color=TRUE_C, lw=2.0, label="truth")
    for level, h in enumerate(LEVELS_MM):
        s = [s for s in snaps if s["level"] == level][-1]
        ax.plot(y, mm(s["field"], d), lw=1.1, label=f"after the {h} mm grid", color=plt.cm.viridis(level / 3.5))
    ax.set_ylabel("displacement (mm)")
    ax.set_xlabel("voxel along the phase-encode axis")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.savefig(os.path.join(out, "line_levels.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_unwarp(d, snaps, out):
    """The corrected line from the fitted field: inverting the push against pulling back."""
    f = snaps[-1]["field"]
    disp = f * d["step"]
    g = snaps[-1]["gain"]  # put the acquired intensities back on the reference's scale
    inv = lm.unwarp(d["obs"], disp) / g
    pb = lm.pullback(d["obs"], disp) / g
    plain = lm.pull(d["obs"], disp) / g
    y = np.arange(len(inv))
    fig, ax = plt.subplots(figsize=(10, 3.4))
    shade_brain(ax, d)
    ax.plot(y, d["ref"], color=REF_C, lw=2.0, label="undistorted reference")
    ax.plot(y, plain, ":", color="0.5", lw=1.3, label="resampled only (what the exported warp does)")
    ax.plot(y, pb, "--", color=FIT_C, lw=1.3, label="resampled × Jacobian")
    ax.plot(y, inv, color=PRED_C, lw=1.4, label="inverse of the push (corrected.nii.gz)")
    ax.set_xlim(*SEG)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_xlabel("voxel along the phase-encode axis")
    fig.savefig(os.path.join(out, "line_unwarp.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def animation(d, snaps, out):
    """The whole fit as a GIF: prediction converging on the acquisition, field on the truth."""
    keep = [s for s in snaps if s["it"] <= 12 or s["it"] % 6 == 0]
    y = np.arange(len(d["obs"]))
    true_mm = mm(d["field_true"], d)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True, gridspec_kw=dict(hspace=0.3))
    for ax in axes:
        shade_brain(ax, d)
    axes[0].plot(y, d["obs"], color=OBS_C, lw=1.3, label="acquired b=0")
    (pred,) = axes[0].plot(y, keep[0]["pred"], "--", color=PRED_C, lw=1.3, label="prediction")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    axes[1].plot(y, true_mm, color=TRUE_C, lw=1.3, label="true displacement (mm)")
    (fld,) = axes[1].plot(y, mm(keep[0]["field"], d), color=FIT_C, lw=1.5, label="fitted")
    axes[1].set_ylim(true_mm.min() - 2, true_mm.max() + 2)
    axes[1].legend(fontsize=8, frameon=False, loc="lower left")
    title = axes[0].set_title("", fontsize=9, loc="left")

    def draw(k):
        s = keep[k]
        pred.set_ydata(s["pred"])
        fld.set_ydata(mm(s["field"], d))
        title.set_text(f"{s['h_mm']} mm control grid, iteration {s['it']}")
        return pred, fld, title

    frames = list(range(len(keep))) + [len(keep) - 1] * 12
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(os.path.join(out, "line_fit.gif"), writer=PillowWriter(fps=8), dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testdata", default="testdata/rpe")
    p.add_argument("--out", default="docs/_static")
    args = p.parse_args()
    diffs = lm.check(args.testdata)
    print("line operators vs the prototype dumps (max abs difference):", diffs)
    assert max(diffs.values()) < 1e-4, "the PyTorch line model no longer matches the prototype"
    d = load_line(args.testdata)
    snaps = fit(d)
    fig_setup(d, args.out)
    fig_cells(d, args.out)
    pp = fig_push_pull(d, args.out)
    fig_gradient(d, args.out)
    fig_steps(d, snaps, args.out)
    fig_levels(d, snaps, args.out)
    fig_unwarp(d, snaps, args.out)
    animation(d, snaps, args.out)
    true_mm = mm(d["field_true"], d)
    for level, h in enumerate(LEVELS_MM):
        s = [s for s in snaps if s["level"] == level][-1]
        err = ((mm(s["field"], d) - true_mm)[d["brain"]] ** 2).mean().sqrt().item()
        print(f"{h:>2} mm grid: data {s['data']:.5f} bending {s['bend']:.4f} gain {s['gain']:.3f} rms error {err:.2f} mm")
    print("at the true field, rms residual push vs acquired / pull vs reference:", pp)


if __name__ == "__main__":
    main()
