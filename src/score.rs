//! Field accuracy against a known displacement (voxels along PE), inside a mask.
use ndarray::Array3;
use serde::Serialize;

#[derive(Serialize, Debug, Clone)]
pub struct FieldScore {
    pub corr: f64,
    pub slope: f64,
    pub rms_residual_mm: f64,
    pub rms_truth_mm: f64,
    pub slope_demeaned: f64,
    pub rms_residual_demeaned_mm: f64,
    pub mean_est_mm: f64,
    pub mean_truth_mm: f64,
    pub n: usize,
}

pub fn score_field(
    est_vox: &Array3<f32>,
    true_vox: &Array3<f32>,
    mask: &Array3<bool>,
    vox_mm: f32,
) -> FieldScore {
    let mut e = Vec::new();
    let mut t = Vec::new();
    for ((a, b), &m) in est_vox.iter().zip(true_vox.iter()).zip(mask.iter()) {
        if m {
            e.push(*a as f64 * vox_mm as f64);
            t.push(*b as f64 * vox_mm as f64);
        }
    }
    let n = e.len();
    let me = e.iter().sum::<f64>() / n as f64;
    let mt = t.iter().sum::<f64>() / n as f64;
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
    let ed: Vec<f64> = e.iter().map(|v| v - me).collect();
    let td: Vec<f64> = t.iter().map(|v| v - mt).collect();
    let corr = dot(&ed, &td) / (dot(&ed, &ed) * dot(&td, &td)).sqrt();
    let rms = |x: &[f64], y: &[f64]| {
        (x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n as f64).sqrt()
    };
    FieldScore {
        corr,
        slope: dot(&e, &t) / dot(&t, &t),
        rms_residual_mm: rms(&e, &t),
        rms_truth_mm: (dot(&t, &t) / n as f64).sqrt(),
        slope_demeaned: dot(&ed, &td) / dot(&td, &td),
        rms_residual_demeaned_mm: rms(&ed, &td),
        mean_est_mm: me,
        mean_truth_mm: mt,
        n,
    }
}
