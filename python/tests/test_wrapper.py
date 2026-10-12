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
