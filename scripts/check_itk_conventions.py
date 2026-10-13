#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check unpile's transform conventions against ANTs itself: LPS/RAS, direction and composition.

    UNPILE_FIXTURES=testdata/fixtures python scripts/check_itk_conventions.py .

Needs ANTs (antsApplyTransforms, antsApplyTransformsToPoints, antsRegistration, ConvertTransformFile
on PATH or in $ANTSPATH), the phasediff fixture, a release build, and nibabel, scipy, nitransforms.
Not run in CI, which has no ANTs; python/tests/test_transforms.py pins the same conventions there.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import nibabel as nb
import numpy as np

REPO = sys.argv[1]
ANTS = os.environ.get("ANTSPATH") or os.path.dirname(shutil.which("antsApplyTransforms") or "")
if not os.path.exists(os.path.join(ANTS, "antsApplyTransforms")):
    sys.exit("ANTs not found: put it on PATH or set ANTSPATH")
sys.path.insert(0, os.path.join(REPO, "python"))
os.environ["UNPILE_BIN"] = os.path.join(REPO, "target", "release", "unpile")
import unpile  # noqa: E402

FX = os.path.join(os.environ.get("UNPILE_FIXTURES", os.path.join(REPO, "testdata", "fixtures")), "phasediff", "unpile")
tmp = tempfile.mkdtemp(prefix="xfmcheck-")
fails = []


def ants(*args):
    subprocess.run([os.path.join(ANTS, args[0]), *args[1:]], check=True, capture_output=True)


def check(name, ok, detail):
    print(("ok   " if ok else "FAIL ") + name + ": " + detail)
    if not ok:
        fails.append(name)


# A transform written by hand in ITK's own terms: LPS, rotation 10 deg about z, translation, and a
# non-zero centre of rotation (FixedParameters), which ITK folds into the offset.
th = np.deg2rad(10)
M = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
t, c = np.array([3.0, -2.0, 5.0]), np.array([10.0, -20.0, 30.0])
txt = os.path.join(tmp, "T.txt")
open(txt, "w").write(
    "#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\n"
    "Parameters: " + " ".join(f"{v:.10f}" for v in list(M.ravel()) + list(t)) + "\n"
    f"FixedParameters: {c[0]} {c[1]} {c[2]}\n")

def ants_ras(path):
    """ANTs' own RAS homogeneous matrix of a transform file."""
    out = os.path.join(tmp, "hm_" + os.path.basename(path) + ".mat")
    ants("ConvertTransformFile", "3", path, out, "--homogeneousMatrix", "--RAS")
    return np.loadtxt(out)


# A. reading the hand-written .txt (non-zero centre) against ANTs' conversion to RAS
ras_txt = unpile.read_itk_affine(txt)
ref_hm = ants_ras(txt)
check("A .txt read = ConvertTransformFile --hm --RAS", np.allclose(ras_txt, ref_hm, atol=1e-5), f"max diff {np.abs(ras_txt - ref_hm).max():.2e}")

# a binary .mat written by antsRegistration itself: rigid fit of a moved copy of the b=0
b0p = os.path.join(FX, "b0_AP.nii.gz")
moved = os.path.join(tmp, "b0_moved.nii.gz")
ants("antsApplyTransforms", "-d", "3", "-i", b0p, "-r", b0p, "-t", txt, "-n", "Linear", "-o", moved)
ants("antsRegistration", "-d", "3", "-o", os.path.join(tmp, "reg_"), "-r", f"[{b0p},{moved},1]",
     "-t", "Rigid[0.1]", "-m", f"MI[{b0p},{moved},1,32]", "-c", "[50x20,1e-6,10]", "-f", "2x1", "-s", "1x0vox")
mat = os.path.join(tmp, "reg_0GenericAffine.mat")
ras_mat = unpile.read_itk_affine(mat)
# moved(x) = b0(T(x)); registering moved (moving) to b0 (fixed) finds R with b0(x) = moved(R(x)), so
# R = inverse(T) if the registration output maps fixed points to moving points
inv_t = np.linalg.inv(ras_txt)
dev = max(np.abs(ras_mat[:3, :3] - inv_t[:3, :3]).max() * 100, np.abs(ras_mat[:3, 3] - inv_t[:3, 3]).max())
check("A antsRegistration output maps fixed (b=0) points to moving points", dev < 0.5, f"vs inverse of the applied transform: max diff {dev:.2f} (mm, or matrix x 100)")
mat_hm = ants_ras(mat)
check("A ANTs binary .mat read = ConvertTransformFile --hm --RAS", np.allclose(ras_mat, mat_hm, atol=1e-5), f"max diff {np.abs(ras_mat - mat_hm).max():.2e}")

# A'. nitransforms (independent ITK reader) on the same files
import nitransforms as nt  # noqa: E402

for path, ours, label in ((txt, ras_txt, ".txt"), (mat, ras_mat, "ANTs .mat")):
    ref_nt = nt.linear.load(path, fmt="itk").matrix
    check(f"A' {label} read = nitransforms", np.allclose(ours, ref_nt, atol=1e-5), f"max diff {np.abs(ours - ref_nt).max():.2e}")

# B. direction and LPS, from ANTs' own point mapping and image resampling
pts_lps = np.array([[0.0, 0.0, 0.0], [12.0, -7.0, 33.0], [-40.0, 25.0, -10.0]])
csv_in, csv_out = os.path.join(tmp, "p.csv"), os.path.join(tmp, "q.csv")
np.savetxt(csv_in, np.c_[pts_lps, np.zeros(3)], delimiter=",", header="x,y,z,t", comments="")
ants("antsApplyTransformsToPoints", "-d", "3", "-i", csv_in, "-o", csv_out, "-t", txt)
ants_pts = np.loadtxt(csv_out, delimiter=",", skiprows=1)[:, :3]
flip = np.array([-1.0, -1.0, 1.0])
ours = np.array([(ras_txt @ np.r_[p * flip, 1])[:3] * flip for p in pts_lps])
check("B antsApplyTransformsToPoints = our map of LPS points", np.allclose(ours, ants_pts, atol=1e-4), f"max diff {np.abs(ours - ants_pts).max():.2e} mm")

# image resampling: a blob at moving-world q lands at fixed-world x with T(x) = q (the pull convention)
grid = nb.load(os.path.join(FX, "b0_AP.nii.gz"))
blob = np.zeros(grid.shape[:3], np.float32)
q_vox = np.array([20, 50, 20])
blob[tuple(q_vox)] = 1000
from scipy.ndimage import gaussian_filter  # noqa: E402

blob = gaussian_filter(blob, 2)
nb.Nifti1Image(blob, grid.affine).to_filename(os.path.join(tmp, "blob.nii.gz"))
ants("antsApplyTransforms", "-d", "3", "-i", os.path.join(tmp, "blob.nii.gz"), "-r", os.path.join(FX, "b0_AP.nii.gz"), "-t", txt, "-n", "Linear", "-o", os.path.join(tmp, "blob_ants.nii.gz"))
out = np.asarray(nb.load(os.path.join(tmp, "blob_ants.nii.gz")).dataobj)
com_vox = np.array([np.sum(np.indices(out.shape)[k] * out) / out.sum() for k in range(3)])
x_ras = (grid.affine @ np.r_[com_vox, 1])[:3]
q_ras = (grid.affine @ np.r_[q_vox, 1])[:3]
mapped = (ras_txt @ np.r_[x_ras, 1])[:3]
check("B antsApplyTransforms reads output x from input at T(x)", np.linalg.norm(mapped - q_ras) < 0.3, f"|T(x) - q| = {np.linalg.norm(mapped - q_ras):.3f} mm")

# write: our .txt (centre 0) resamples exactly like the original (non-zero centre)
ours_txt = unpile.write_itk_affine(os.path.join(tmp, "T_ours.txt"), ras_txt)
ants("antsApplyTransforms", "-d", "3", "-i", os.path.join(tmp, "blob.nii.gz"), "-r", os.path.join(FX, "b0_AP.nii.gz"), "-t", str(ours_txt), "-n", "Linear", "-o", os.path.join(tmp, "blob_ours.nii.gz"))
d = np.abs(np.asarray(nb.load(os.path.join(tmp, "blob_ours.nii.gz")).dataobj) - out).max()
check("A write_itk_affine resamples identically in ANTs", d < 1e-3, f"max diff {d:.2e}")

# C. prepare(reference_xfm=...) against antsApplyTransforms with the same file
ref = os.path.join(FX, "T2w.nii.gz")
prep = unpile.prepare(os.path.join(FX, "b0_AP.nii.gz"), os.path.join(tmp, "prep"), mask=os.path.join(FX, "brain_mask.nii.gz"), reference=ref, reference_xfm=mat, pe_dir="j-")
ants("antsApplyTransforms", "-d", "3", "-i", ref, "-r", os.path.join(FX, "b0_AP.nii.gz"), "-t", mat, "-n", "BSpline", "-o", os.path.join(tmp, "ref_ants.nii.gz"))
a = np.asarray(nb.load(str(prep["reference"])).dataobj, float)
b = np.asarray(nb.load(os.path.join(tmp, "ref_ants.nii.gz")).dataobj, float)
brain = np.asarray(nb.load(os.path.join(FX, "brain_mask.nii.gz")).dataobj) > 0
r = np.corrcoef(a[brain], b[brain])[0, 1]
check("C prepare(reference_xfm) matches antsApplyTransforms", r > 0.995, f"corr {r:.4f} in the brain")

# D/E. a fit with --pose after the initial transform
truth = os.path.join(FX, json.load(open(os.path.join(FX, "meta.json")))["truth_xfm"]["T2w"])
meta = json.load(open(os.path.join(FX, "meta.json")))
res = unpile.fit(prep["epi"], prep["mask"], os.path.join(tmp, "fit"), reference=prep["reference"], metric="lncc", pose=True, pe_dir=meta["AP"]["pe"], total_readout_time=meta["AP"]["trt"], levels=[24, 12, 6], iters=20, reference_init_xfm=prep["reference_xfm"], truth_xfm=truth, extra=["--window", "5", "--lam", "1"])
# D. unpile's own resampling of the reference under its pose = ANTs with pose_xfm.txt
ants("antsApplyTransforms", "-d", "3", "-i", str(prep["reference"]), "-r", str(prep["epi"]), "-t", str(res.pose_xfm), "-n", "Linear", "-o", os.path.join(tmp, "posed_ants.nii.gz"))
a = np.asarray(nb.load(str(res.reference_posed)).dataobj, float)
b = np.asarray(nb.load(os.path.join(tmp, "posed_ants.nii.gz")).dataobj, float)
r = np.corrcoef(a[brain], b[brain])[0, 1]
check("D reference_posed = antsApplyTransforms -t pose_xfm.txt", r > 0.995, f"corr {r:.4f}")
# E. the composed file = ANTs chaining -t init -t pose, from the original reference
# ANTs: `-t A -t B` maps a point x to B(A(x)) (checked below with points), so init o pose is -t pose -t init
ants("antsApplyTransforms", "-d", "3", "-i", ref, "-r", str(prep["epi"]), "-t", str(res.pose_xfm), "-t", str(prep["reference_xfm"]), "-n", "Linear", "-o", os.path.join(tmp, "chain.nii.gz"))
ants("antsApplyTransforms", "-d", "3", "-i", ref, "-r", str(prep["epi"]), "-t", str(res.pose_composed_xfm), "-n", "Linear", "-o", os.path.join(tmp, "composed.nii.gz"))
a = np.asarray(nb.load(os.path.join(tmp, "chain.nii.gz")).dataobj, float)
b = np.asarray(nb.load(os.path.join(tmp, "composed.nii.gz")).dataobj, float)
check("E pose_composed_xfm.txt = ANTs -t pose -t init", np.abs(a - b).max() < 1e-2 * np.abs(a).max(), f"max diff {np.abs(a - b).max():.2e} (image max {np.abs(a).max():.0f})")
Pm, Im, Cm = (unpile.read_itk_affine(str(f)) for f in (res.pose_xfm, prep["reference_xfm"], res.pose_composed_xfm))
check("E composed matrix = init @ pose", np.allclose(Cm, Im @ Pm, atol=1e-6), f"max diff {np.abs(Cm - Im @ Pm).max():.2e}")
pts = np.array([[0.0, 0, 0], [30, -40, 20], [-25, 10, -35]])
np.savetxt(os.path.join(tmp, "pc.csv"), np.c_[pts * flip, np.zeros(3)], delimiter=",", header="x,y,z,t", comments="")
ants("antsApplyTransformsToPoints", "-d", "3", "-i", os.path.join(tmp, "pc.csv"), "-o", os.path.join(tmp, "qc.csv"), "-t", str(res.pose_xfm), "-t", str(prep["reference_xfm"]))
qc = np.loadtxt(os.path.join(tmp, "qc.csv"), delimiter=",", skiprows=1)[:, :3] * flip
d = np.abs(qc - np.array([(Im @ Pm @ np.r_[p, 1])[:3] for p in pts])).max()
check("E ANTs points through -t pose -t init = init(pose(x))", d < 1e-3, f"max diff {d:.2e} mm")
s = res.score["pose_vs_truth"]
check("E composed pose vs the recorded truth", s["rotation_deg"] < 3 and s["translation_mm"] < 3, f"{s['rotation_deg']:.2f} deg / {s['translation_mm']:.2f} mm (identity {s['identity_rotation_deg']:.2f} / {s['identity_translation_mm']:.2f})")
print("FAILED:" if fails else "all checks passed", fails if fails else "")
sys.exit(1 if fails else 0)
