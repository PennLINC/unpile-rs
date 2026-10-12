//! Truth-scored runs on TRXScan fixtures. Needs `UNPILE_FIXTURES` pointing at the directory that
//! `scripts/make_fixtures.py` filled; without it every test here prints a note and passes.
//! Each assertion goes through `expect`, whose message names the quantity, its value and the bound.
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn fixture(name: &str) -> Option<PathBuf> {
    let root = std::env::var_os("UNPILE_FIXTURES").map(PathBuf::from)?;
    let dir = root.join(name).join("unpile");
    if dir.join("meta.json").exists() {
        Some(dir)
    } else {
        eprintln!(
            "fixture {name} not found under {}; skipping",
            root.display()
        );
        None
    }
}

fn json(path: &Path) -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap()
}

/// Run `unpile fit` with the CI schedule (levels 24/12/6 mm, 20 iterations) and return score.json.
fn fit(dir: &Path, out: &str, extra: &[&str]) -> Value {
    let meta = json(&dir.join("meta.json"));
    let ap = &meta["AP"];
    let out_dir = dir.join("out").join(out);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_unpile"));
    cmd.args(["--device", "flex", "fit"])
        .arg("--epi")
        .arg(dir.join("b0_AP.nii.gz"))
        .arg("--mask")
        .arg(dir.join("brain_mask.nii.gz"))
        .arg("--total-readout-time")
        .arg(ap["trt"].as_f64().unwrap().to_string())
        .arg("--pe-dir")
        .arg(ap["pe"].as_str().unwrap())
        .args(["--levels", "24,12,6", "--iters", "20"])
        .arg("--truth-field")
        .arg(dir.join("truth_field.nii.gz"))
        .arg("--out")
        .arg(&out_dir)
        .args(extra);
    let status = cmd.status().expect("unpile binary");
    assert!(status.success(), "unpile fit failed for {out}");
    json(&out_dir.join("score.json"))
}

fn expect(fails: &mut Vec<String>, what: &str, value: f64, lo: f64, hi: f64) {
    if value >= lo && value <= hi {
        println!("  ok   {what} = {value:.3} within [{lo}, {hi}]");
    } else {
        fails.push(format!("{what} = {value:.3}, expected within [{lo}, {hi}]"));
    }
}

fn finish(fails: Vec<String>) {
    assert!(fails.is_empty(), "\n{}", fails.join("\n"));
}

/// GRE refinement: a fieldmap acquired after a recorded 5 degree / 4 mm movement is brought onto the
/// EPI by its pose, shim and residual terms; and the clean-reference ceiling of the same fixture.
#[test]
fn phasediff_gre_refinement() {
    let Some(dir) = fixture("phasediff") else {
        return;
    };
    let meta = json(&dir.join("meta.json"));
    let truth_xfm = dir.join(
        meta["truth_xfm"]["fmap"]
            .as_str()
            .expect("fmap truth transform"),
    );
    let mut fails = Vec::new();
    let s = fit(
        &dir,
        "clean",
        &[
            "--reference",
            dir.join("clean_b0.nii.gz").to_str().unwrap(),
            "--lam-mm",
            "5",
        ],
    );
    expect(
        &mut fails,
        "phasediff clean reference: field corr",
        s["field"]["corr"].as_f64().unwrap(),
        0.98,
        1.0,
    );
    expect(
        &mut fails,
        "phasediff clean reference: field rms residual mm",
        s["field"]["rms_residual_mm"].as_f64().unwrap(),
        0.0,
        0.8,
    );
    let s = fit(
        &dir,
        "gre",
        &[
            "--reference",
            dir.join("clean_b0.nii.gz").to_str().unwrap(),
            "--lam-mm",
            "5",
            "--fieldmap",
            dir.join("gre_hz.nii.gz").to_str().unwrap(),
            "--fieldmap-refine",
            "pose,shim,delta",
            "--truth-xfm",
            truth_xfm.to_str().unwrap(),
        ],
    );
    expect(
        &mut fails,
        "phasediff GRE prior: field corr",
        s["field"]["corr"].as_f64().unwrap(),
        0.98,
        1.0,
    );
    expect(
        &mut fails,
        "phasediff GRE prior: field rms residual mm",
        s["field"]["rms_residual_mm"].as_f64().unwrap(),
        0.0,
        0.6,
    );
    let id_rot = s["gre_pose_vs_truth"]["identity_rotation_deg"]
        .as_f64()
        .unwrap();
    expect(
        &mut fails,
        "phasediff GRE prior: recorded fieldmap offset deg (fixture sanity)",
        id_rot,
        4.0,
        7.0,
    );
    expect(
        &mut fails,
        "phasediff GRE prior: pose residual rotation deg",
        s["gre_pose_vs_truth"]["rotation_deg"].as_f64().unwrap(),
        0.0,
        3.0,
    );
    expect(
        &mut fails,
        "phasediff GRE prior: pose residual translation mm",
        s["gre_pose_vs_truth"]["translation_mm"].as_f64().unwrap(),
        0.0,
        1.5,
    );
    finish(fails);
}

/// Fieldmap-less: the T2w acquired after a recorded 7 degree / 5 mm movement, LNCC with a joint pose.
#[test]
fn phasediff_t2w_fieldmapless() {
    let Some(dir) = fixture("phasediff") else {
        return;
    };
    let meta = json(&dir.join("meta.json"));
    let truth_xfm = dir.join(
        meta["truth_xfm"]["T2w"]
            .as_str()
            .expect("T2w truth transform"),
    );
    let mut fails = Vec::new();
    let s = fit(
        &dir,
        "t2w",
        &[
            "--reference",
            dir.join("T2w.nii.gz").to_str().unwrap(),
            "--metric",
            "lncc",
            "--window",
            "5",
            "--pose",
            "--lam",
            "1",
            "--truth-xfm",
            truth_xfm.to_str().unwrap(),
        ],
    );
    expect(
        &mut fails,
        "phasediff T2w: recorded anatomical offset deg (fixture sanity)",
        s["pose_vs_truth"]["identity_rotation_deg"]
            .as_f64()
            .unwrap(),
        5.0,
        9.0,
    );
    expect(
        &mut fails,
        "phasediff T2w: pose residual rotation deg",
        s["pose_vs_truth"]["rotation_deg"].as_f64().unwrap(),
        0.0,
        3.0,
    );
    expect(
        &mut fails,
        "phasediff T2w: pose residual translation mm",
        s["pose_vs_truth"]["translation_mm"].as_f64().unwrap(),
        0.0,
        3.0,
    );
    expect(
        &mut fails,
        "phasediff T2w: field corr (loose: a refiner, not a fieldmap)",
        s["field"]["corr"].as_f64().unwrap(),
        0.3,
        1.0,
    );
    finish(fails);
}

/// Two opposite-polarity images, no reference.
#[test]
fn rpe_two_blip() {
    let Some(dir) = fixture("rpe") else { return };
    let mut fails = Vec::new();
    let s = fit(
        &dir,
        "pepolar",
        &[
            "--reverse-epi",
            dir.join("b0_PA.nii.gz").to_str().unwrap(),
            "--lam-mm",
            "5",
        ],
    );
    expect(
        &mut fails,
        "rpe two-blip: field corr",
        s["field"]["corr"].as_f64().unwrap(),
        0.97,
        1.0,
    );
    expect(
        &mut fails,
        "rpe two-blip: field slope",
        s["field"]["slope"].as_f64().unwrap(),
        0.85,
        1.1,
    );
    finish(fails);
}
