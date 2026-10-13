"""ITK transform files: reading, writing and the initial reference transform.

ITK (and so ANTs) stores points in LPS; unpile works in RAS. These tests pin the conversion and the
direction (a map from a point of the fixed EPI grid to the point read in the moving image) against
ITK's own parameterisation and, when installed, nitransforms. The conventions were also checked
against antsApplyTransforms and antsApplyTransformsToPoints directly; ANTs is not available in CI.
"""

import json
import os
from pathlib import Path

import pytest

import unpile

np = pytest.importorskip("numpy")
FIXTURES = os.environ.get("UNPILE_FIXTURES")


def _itk_txt(path, m_lps, t_lps, centre_lps):
    path.write_text(
        "#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\n"
        "Parameters: " + " ".join(f"{v:.10f}" for v in list(np.ravel(m_lps)) + list(t_lps)) + "\n"
        "FixedParameters: " + " ".join(str(v) for v in centre_lps) + "\n"
    )
    return path


def _example(tmp_path):
    th = np.deg2rad(10)
    m = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    t, c = np.array([3.0, -2.0, 5.0]), np.array([10.0, -20.0, 30.0])
    return m, t, c, _itk_txt(tmp_path / "T.txt", m, t, c)


def test_read_matches_itk_parameterisation(tmp_path):
    m, t, c, path = _example(tmp_path)
    ras = unpile.read_itk_affine(path)
    flip = np.array([-1.0, -1.0, 1.0])
    for p_lps in ([0.0, 0, 0], [12.0, -7.0, 33.0], [-40.0, 25.0, -10.0]):
        p_lps = np.array(p_lps)
        expected_lps = m @ (p_lps - c) + c + t  # ITK's MatrixOffsetTransformBase
        got = (ras @ np.r_[p_lps * flip, 1])[:3] * flip
        assert np.allclose(got, expected_lps, atol=1e-8)


def test_read_matches_nitransforms(tmp_path):
    nt = pytest.importorskip("nitransforms")
    sio = pytest.importorskip("scipy.io")
    m, t, c, path = _example(tmp_path)
    assert np.allclose(unpile.read_itk_affine(path), nt.linear.load(str(path), fmt="itk").matrix, atol=1e-6)
    # the ANTs binary layout (MATLAB v4): 12 parameters and the centre
    mat = tmp_path / "T.mat"
    sio.savemat(str(mat), {"AffineTransform_double_3_3": np.r_[m.ravel(), t][:, None], "fixed": c[:, None]}, format="4")
    assert np.allclose(unpile.read_itk_affine(mat), nt.linear.load(str(mat), fmt="itk").matrix, atol=1e-6)


def test_write_read_roundtrip(tmp_path):
    _, _, _, path = _example(tmp_path)
    ras = unpile.read_itk_affine(path)
    again = unpile.read_itk_affine(unpile.write_itk_affine(tmp_path / "again.txt", ras))
    assert np.allclose(ras, again, atol=1e-8)


def test_rejects_non_affine(tmp_path):
    p = tmp_path / "warp.txt"
    p.write_text("#Insight Transform File V1.0\n#Transform 0\nTransform: BSplineTransform_double_3_3\nParameters: 1 2\nFixedParameters: 0\n")
    with pytest.raises(ValueError):
        unpile.read_itk_affine(p)


@pytest.mark.skipif(FIXTURES is None, reason="needs the TRXScan fixtures")
def test_prepare_reference_xfm(tmp_path):
    nb = pytest.importorskip("nibabel")
    d = Path(FIXTURES) / "phasediff" / "unpile"
    plain = unpile.prepare(d / "b0_AP.nii.gz", tmp_path / "plain", mask=d / "brain_mask.nii.gz", reference=d / "T2w.nii.gz", pe_dir="j-")
    ident = _itk_txt(tmp_path / "id.txt", np.eye(3), np.zeros(3), np.zeros(3))
    same = unpile.prepare(d / "b0_AP.nii.gz", tmp_path / "same", mask=d / "brain_mask.nii.gz", reference=d / "T2w.nii.gz", pe_dir="j-", reference_xfm=ident)
    a = np.asarray(nb.load(str(plain["reference"])).dataobj)
    assert np.allclose(a, np.asarray(nb.load(str(same["reference"])).dataobj), atol=1e-3 * np.abs(a).max())
    # a translation of one voxel along the first axis: output x reads the reference at x + one voxel
    epi = nb.load(str(d / "b0_AP.nii.gz"))
    step_ras = epi.affine[:3, 0]
    shift = _itk_txt(tmp_path / "shift.txt", np.eye(3), step_ras * np.array([-1.0, -1.0, 1.0]), np.zeros(3))
    moved = unpile.prepare(d / "b0_AP.nii.gz", tmp_path / "moved", mask=d / "brain_mask.nii.gz", reference=d / "T2w.nii.gz", pe_dir="j-", reference_xfm=shift)
    b = np.asarray(nb.load(str(moved["reference"])).dataobj)
    assert np.allclose(b[:-1], a[1:], atol=1e-3 * np.abs(a).max())
    assert Path(moved["reference_xfm"]).exists()


@pytest.mark.skipif(FIXTURES is None, reason="needs the TRXScan fixtures")
def test_fit_composes_the_initial_transform(tmp_path):
    nb = pytest.importorskip("nibabel")
    d = Path(FIXTURES) / "phasediff" / "unpile"
    meta = json.load(open(d / "meta.json"))
    th = np.deg2rad(3)
    init = _itk_txt(tmp_path / "init.txt", np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]), np.array([2.0, -1.0, 3.0]), np.array([5.0, 5.0, 5.0]))
    prep = unpile.prepare(d / "b0_AP.nii.gz", tmp_path / "prep", mask=d / "brain_mask.nii.gz", reference=d / "T2w.nii.gz", pe_dir=meta["AP"]["pe"], reference_xfm=init)
    r = unpile.fit(prep["epi"], prep["mask"], tmp_path / "fit", reference=prep["reference"], metric="lncc", pose=True, pe_dir=meta["AP"]["pe"],
                   total_readout_time=meta["AP"]["trt"], levels=[24, 12, 6], iters=20, reference_init_xfm=prep["reference_xfm"],
                   truth_xfm=d / meta["truth_xfm"]["T2w"], extra=["--window", "5", "--lam", "1"])
    assert r.reference_posed is not None and r.pose_composed_xfm is not None
    assert nb.load(str(r.reference_posed)).shape == nb.load(str(prep["epi"])).shape
    # init o pose: a point of the EPI grid goes through the fitted pose, then the initial transform
    composed = unpile.read_itk_affine(r.pose_composed_xfm)
    assert np.allclose(composed, unpile.read_itk_affine(prep["reference_xfm"]) @ unpile.read_itk_affine(r.pose_xfm), atol=1e-6)
    # --truth-xfm scores the composition against the original reference's recorded transform
    assert r.score["pose_vs_truth"]["rotation_deg"] < 3 and r.score["pose_vs_truth"]["translation_mm"] < 3
