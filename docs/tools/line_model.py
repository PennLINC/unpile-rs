# SPDX-License-Identifier: MIT OR Apache-2.0
"""unpile's forward model on one phase-encode line, in PyTorch.

These functions are the one-dimensional versions of the operators in ``src/ops.rs`` and
``src/field.rs``. They work on tensors whose last axis is the phase-encode axis, so a single line
is a tensor of shape ``[n]`` and a whole volume (PE last) is ``[n0, n1, n]``. ``check()`` runs
them on every line of ``testdata/rpe`` and compares with the PyTorch prototype's stored outputs,
the same dumps ``unpile check`` holds the Rust operators to.
"""

import json
import math
import os

import numpy as np
import torch

SCALE = 100.0  # B-spline coefficients are stored in units of 100 Hz, as in src/field.rs


def ramp(x, membership="tent", sigma=0.5):
    """How much of the half-line (-inf, x] lies inside a voxel boundary, once the boundary is blurred.

    With a sharp boundary this is max(x, 0). The blur makes it smooth, so the overlap of a moved
    cell with a voxel changes smoothly as the cell moves, and has a gradient.
    """
    if membership == "sharp":  # exact overlap with unblurred voxels: for worked examples only
        return x.clamp_min(0)
    if membership == "tent":
        h = 0.5
        u = x.clamp(-h, h) + h
        return u * u / (4 * h) + (x - h).clamp_min(0)
    z = x / sigma
    return sigma * (z * 0.5 * (1 + torch.erf(z / math.sqrt(2))) + torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi))


def moved_cells(disp):
    """Where each voxel of the undistorted image ends up: the interval [a_i, b_i] in voxels.

    Voxel i is the cell [i - 1/2, i + 1/2]. Each edge moves by the displacement at that edge (the
    mean of the two voxels it separates), so neighbouring cells stay glued together.
    """
    n = disp.shape[-1]
    edge_disp = torch.cat([disp[..., :1], 0.5 * (disp[..., 1:] + disp[..., :-1]), disp[..., -1:]], -1)
    edges = torch.arange(n + 1, dtype=disp.dtype) - 0.5 + edge_disp
    lo, hi = edges[..., :-1], edges[..., 1:]
    return torch.minimum(lo, hi), torch.maximum(lo, hi)  # a fold flips a cell; it still has a length


def splat_weights(disp, membership="tent", sigma=0.5, nmax=10):
    """The pushforward as a matrix: W[..., i, j] = share of cell i that lands in voxel j.

    Every row sums to one (inside the field of view), which is what conserves the signal.
    """
    n = disp.shape[-1]
    a, b = moved_cells(disp)
    length = (b - a).clamp_min(0.02)
    j = torch.arange(n, dtype=disp.dtype)
    A, B = a[..., :, None], b[..., :, None]
    lo, hi = j - 0.5, j + 0.5

    def r(x):
        return ramp(x, membership, sigma)

    overlap = (r(B - lo) - r(B - hi)) - (r(A - lo) - r(A - hi))
    w = overlap / length[..., :, None]
    # as in src/ops.rs, a cell only reaches the nm voxels starting one before the one it lands in
    reach = math.ceil(length.max().item()) + 2 + math.ceil(4 * sigma)
    nm = max(1, min(nmax, reach))
    offset = j - ((a + 0.5).floor() - 1)[..., :, None]
    return w * ((offset >= 0) & (offset < nm))


def splat(img, disp, **kw):
    """Mass-conserving pushforward: spread each voxel's signal evenly over its moved cell."""
    return torch.einsum("...i,...ij->...j", img, splat_weights(disp, **kw))


def ddx(d):
    """Central difference along the last axis, one-sided at the ends."""
    return torch.cat([d[..., 1:2] - d[..., :1], 0.5 * (d[..., 2:] - d[..., :-2]), d[..., -1:] - d[..., -2:-1]], -1)


def pull(img, disp):
    """Linear interpolation of ``img`` at y + d(y), fading to zero one voxel past either end."""
    n = img.shape[-1]
    padded = torch.nn.functional.pad(img, (1, 1))
    p = (torch.arange(n, dtype=img.dtype) + disp + 1).clamp(0, n + 1)
    i0 = p.floor().clamp(0, n)
    w = p - i0
    i0 = i0.long()
    return padded.gather(-1, i0) * (1 - w) + padded.gather(-1, i0 + 1) * w


def pullback(img, disp):
    """TOPUP-style unwarp of an observation: resample at y + d(y), times the Jacobian 1 + d'(y)."""
    return pull(img, disp) * (1 + ddx(disp)).clamp_min(0.05)


def gaussian_smooth(x, sigma):
    """Gaussian smoothing along the last axis with replicate ends; identity below 0.3 voxel."""
    if sigma <= 0.3:
        return x
    r = math.ceil(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    xp = torch.nn.functional.pad(x[None, None], (r, r), mode="replicate")
    return torch.nn.functional.conv1d(xp, k[None, None])[0, 0]


def bspline_matrix(n, h):
    """Cubic B-spline basis on n voxels with knots every h voxels: field = M @ coefficients."""
    k = math.ceil((n - 1) / h) + 3
    m = np.zeros((n, k))
    for x in range(n):
        u = x / h
        i = math.floor(u)
        t = u - i
        w = [(1 - t) ** 3 / 6, (3 * t**3 - 6 * t**2 + 4) / 6, (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6, t**3 / 6]
        for jj in range(4):
            m[x, min(max(i + jj, 0), k - 1)] += w[jj]
    return torch.tensor(m, dtype=torch.float32)


def bending_energy(f, spacing_mm):
    """Mean squared second derivative of the field (Hz/mm^2), circular as in src/ops.rs."""
    f2 = (torch.roll(f, -1, -1) - 2 * f + torch.roll(f, 1, -1)) / spacing_mm**2
    return (f2 * f2).mean()


def ssd_with_gain(pred, obs, mask):
    """Squared difference inside the mask after the best overall intensity gain (closed form)."""
    gain = ((mask * pred * obs).sum() / (mask * pred * pred).sum().clamp_min(1e-12)).detach()
    diff = gain * pred - obs
    return (mask * diff * diff).sum() / mask.sum(), gain


def restore(observations, displacements, ridge=1e-3, grad_reg=0.05, **kw):
    """The image that, pushed forward with each displacement, best explains each observation.

    Solves (sum_k W_k W_k^T + ridge I + grad_reg D^T D) x = sum_k W_k obs_k, the normal equations
    that ``unwarp`` (one observation) and ``solve_latent`` (two) in src/problem.rs run conjugate
    gradients on, there with a 3-D gradient penalty.
    """
    n = observations[0].shape[-1]
    dmat = torch.diff(torch.eye(n), dim=0)
    lhs = ridge * torch.eye(n) + grad_reg * dmat.T @ dmat
    rhs = torch.zeros(n)
    for obs, disp in zip(observations, displacements):
        w = splat_weights(disp, **kw)  # rows: source voxels, so obs = w.T @ x
        lhs = lhs + w @ w.T
        rhs = rhs + w @ obs
    return torch.linalg.solve(lhs, rhs)


def unwarp(obs, disp, ridge=1e-3, grad_reg=0.05, **kw):
    """The corrected line from one observation (``corrected.nii.gz``)."""
    return restore([obs], [disp], ridge, grad_reg, **kw)


def check(testdata):
    """Compare with the prototype's dumps on every line of testdata/rpe; return the max differences."""
    import nibabel as nb

    meta = json.load(open(os.path.join(testdata, "meta.json")))
    norm = json.load(open(os.path.join(testdata, "torch_values.json")))["norm"]
    pe = meta["pe_axis"]
    order = [x for x in range(3) if x != pe] + [pe]

    def load(name):
        a = np.asarray(nb.load(os.path.join(testdata, name)).dataobj, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(np.transpose(a, order)))

    disp = load("field_true.nii.gz") * (meta["sign"] * meta["trt"])
    out = {}
    pushed = splat(load("clean.nii.gz") / norm, disp, membership="gauss")
    out["splat"] = (pushed - load("torch_splat_clean_at_truth.nii.gz")).abs().max().item()
    pulled = pullback(load("b0.nii.gz") / norm, disp)
    out["pullback"] = (pulled - load("torch_pullback_b0_at_truth.nii.gz")).abs().max().item()
    return out
