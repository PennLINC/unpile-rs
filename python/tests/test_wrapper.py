import json
import os
from pathlib import Path

import pytest

import unpile

ROOT = Path(__file__).resolve().parents[2]


def test_binary_is_found():
    assert os.path.isfile(unpile.find_unpile_bin())


def test_sidecar_values(tmp_path):
    side = tmp_path / "epi.json"
    side.write_text(json.dumps({"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.0525}))
    assert unpile.sidecar_values(side) == ("j-", 0.0525)
    with pytest.raises(KeyError):
        unpile.sidecar_values({"PhaseEncodingDirection": "j"})


def test_operator_check():
    report = unpile.check(ROOT / "testdata" / "rpe")
    assert report.count("corr 1.000000") >= 6, report


@pytest.mark.skipif("UNPILE_FIXTURES" not in os.environ, reason="needs the TRXScan fixtures")
def test_fit_on_fixture(tmp_path):
    d = Path(os.environ["UNPILE_FIXTURES"]) / "phasediff" / "unpile"
    meta = json.load(open(d / "meta.json"))
    r = unpile.fit(d / "b0_AP.nii.gz", d / "brain_mask.nii.gz", tmp_path / "fit", pe_dir=meta["AP"]["pe"], total_readout_time=meta["AP"]["trt"],
                   reference=d / "clean_b0.nii.gz", lam_mm=5, levels=[24, 12, 6], iters=20, truth_field=d / "truth_field.nii.gz")
    assert r.field_hz.exists() and r.sdc_warp.exists() and r.corrected.exists()
    assert r.score["field"]["corr"] > 0.98


def test_reorient_pe_dir():
    np = pytest.importorskip("numpy")
    pytest.importorskip("nibabel")

    ras = np.diag([2.0, 2.0, 2.0, 1.0])
    las = np.diag([-2.0, 2.0, 2.0, 1.0])
    lps = np.diag([-2.0, -2.0, 2.0, 1.0])
    # a flip of the y axis flips the sign; x flips do not touch a j direction
    assert unpile.reorient_pe_dir("j-", ras, las) == "j-"
    assert unpile.reorient_pe_dir("j-", ras, lps) == "j"
    assert unpile.reorient_pe_dir("j", lps, ras) == "j-"
    # a permutation moves it to the axis that now points along the world axis
    swapped = np.array([[0, 2.0, 0, 0], [2.0, 0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1]])
    assert unpile.reorient_pe_dir("j-", ras, swapped) == "i-"
    assert unpile.reorient_pe_dir("i", swapped, ras) == "j"


@pytest.mark.skipif("UNPILE_FIXTURES" not in os.environ, reason="needs the TRXScan fixtures")
def test_check_polarity_on_fixture(tmp_path):
    d = Path(os.environ["UNPILE_FIXTURES"]) / "phasediff" / "unpile"
    meta = json.load(open(d / "meta.json"))
    r = unpile.check_polarity(d / "b0_AP.nii.gz", d / "brain_mask.nii.gz", d / "clean_b0.nii.gz", d / "gre_hz.nii.gz", meta["AP"]["pe"], meta["AP"]["trt"], work_dir=tmp_path)
    assert r["consistent"], r
    assert r["margin"] > 0.3, r
    with pytest.warns(UserWarning):
        wrong = unpile.check_polarity(d / "b0_AP.nii.gz", d / "brain_mask.nii.gz", d / "clean_b0.nii.gz", d / "gre_hz.nii.gz", "j", meta["AP"]["trt"], work_dir=tmp_path / "wrong")
    assert not wrong["consistent"]


@pytest.mark.skipif("UNPILE_FIXTURES" not in os.environ, reason="needs the TRXScan fixtures")
def test_prepare_refuses_non_las_and_reorient_is_explicit(tmp_path):
    nb = pytest.importorskip("nibabel")
    pytest.importorskip("nilearn")
    d = Path(os.environ["UNPILE_FIXTURES"]) / "phasediff" / "unpile"
    # an RAS copy of the LAS fixture image: prepare must refuse it, reorient must bring it back
    img = nb.load(d / "b0_AP.nii.gz")
    ras = img.as_reoriented(nb.orientations.ornt_transform(nb.orientations.io_orientation(img.affine), nb.orientations.axcodes2ornt(("R", "A", "S"))))
    nb.save(ras, tmp_path / "b0_ras.nii.gz")
    with pytest.warns(UserWarning):
        unpile.warn_if_not_las(tmp_path / "b0_ras.nii.gz")
    with pytest.raises(ValueError, match="RAS\\+, not LAS\\+"):
        unpile.prepare(tmp_path / "b0_ras.nii.gz", tmp_path / "prep", mask=d / "brain_mask.nii.gz", pe_dir="j-")
    las_path, pe_dir = unpile.reorient(tmp_path / "b0_ras.nii.gz", "LAS", pe_dir="j-", out=tmp_path / "b0_las.nii.gz")
    assert unpile.axcodes(las_path) == ("L", "A", "S")
    assert pe_dir == "j-"  # an x flip does not touch a j direction
    p = unpile.prepare(las_path, tmp_path / "prep", mask=d / "brain_mask.nii.gz", pe_dir=pe_dir)
    assert unpile.axcodes(p["epi"]) == ("L", "A", "S") and p["pe_dir"] == "j-"


@pytest.mark.skipif("UNPILE_FIXTURES" not in os.environ, reason="needs the TRXScan fixtures")
def test_preflight_catches_shifted_and_cropped_references(tmp_path):
    nb = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")
    d = Path(os.environ["UNPILE_FIXTURES"]) / "phasediff" / "unpile"
    ok = unpile.preflight(d / "b0_AP.nii.gz", d / "brain_mask.nii.gz", reference=d / "clean_b0.nii.gz", fieldmap=d / "gre_hz.nii.gz", warn=False)
    assert ok["reference"]["com_offset_mm"] < 3 and ok["reference"]["coverage_of_mask"] > 0.95 and not ok["warnings"], ok
    # the same reference with its content shifted 20 mm along y, resampled back onto the grid
    from nilearn.image import resample_to_img

    ref = nb.load(d / "clean_b0.nii.gz")
    moved_aff = ref.affine.copy()
    moved_aff[1, 3] += 20.0
    shifted = resample_to_img(nb.Nifti1Image(np.asarray(ref.dataobj), moved_aff), ref, interpolation="continuous")
    nb.save(shifted, tmp_path / "shifted.nii.gz")
    with pytest.warns(UserWarning, match="centre of mass"):
        r = unpile.preflight(d / "b0_AP.nii.gz", d / "brain_mask.nii.gz", reference=tmp_path / "shifted.nii.gz")
    assert r["reference"]["com_offset_mm"] > 10
    # a reference with the top third of the brain zeroed
    data = np.asarray(ref.dataobj).copy()
    data[:, :, data.shape[2] * 2 // 3 :] = 0
    nb.save(nb.Nifti1Image(data, ref.affine), tmp_path / "cropped.nii.gz")
    with pytest.warns(UserWarning, match="field of view may stop short"):
        r = unpile.preflight(d / "b0_AP.nii.gz", d / "brain_mask.nii.gz", reference=tmp_path / "cropped.nii.gz")
    assert r["reference"]["coverage_of_mask"] < 0.95


@pytest.mark.skipif("UNPILE_FIXTURES" not in os.environ, reason="needs the TRXScan fixtures")
def test_prepare_fov_overlap(tmp_path):
    nb = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")
    pytest.importorskip("nilearn")
    d = Path(os.environ["UNPILE_FIXTURES"]) / "phasediff" / "unpile"
    # a T2w whose header puts it 200 mm away from the EPI: no overlap, prepare must refuse it
    t2 = nb.load(d / "T2w.nii.gz")
    far = t2.affine.copy()
    far[0, 3] += 200.0
    nb.save(nb.Nifti1Image(np.asarray(t2.dataobj), far), tmp_path / "far.nii.gz")
    with pytest.raises(ValueError, match="field of view contains only"):
        unpile.prepare(d / "b0_AP.nii.gz", tmp_path / "prep", mask=d / "brain_mask.nii.gz", reference=tmp_path / "far.nii.gz", pe_dir="j-")
    p = unpile.prepare(d / "b0_AP.nii.gz", tmp_path / "prep2", mask=d / "brain_mask.nii.gz", reference=d / "T2w.nii.gz", pe_dir="j-")
    assert p["reference_fov_overlap"] > 0.99
