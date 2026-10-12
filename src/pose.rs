//! Rigid pose of a reference image: resample it at rotated/translated coordinates with trilinear
//! interpolation (zero outside), differentiable in the six pose parameters through the
//! interpolation weights, like grid_sample. Parametrisation matches the prototype:
//! rotations pose[0..3] * POSE_ROT (rad), translations pose[3..6] * POSE_TRANS (mm), about the
//! grid centre, in the PE-last axes.
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};

pub const POSE_ROT: f32 = 0.05;
pub const POSE_TRANS: f32 = 2.0;

/// Per-axis voxel index grids as float tensors [n0, n1, n2].
pub fn index_grids(shape: [usize; 3], device: &Device) -> [Tensor<3>; 3] {
    let [n0, n1, n2] = shape;
    let mut g = [vec![0f32; n0 * n1 * n2], vec![0f32; n0 * n1 * n2], vec![0f32; n0 * n1 * n2]];
    let mut p = 0;
    for i in 0..n0 {
        for j in 0..n1 {
            for k in 0..n2 {
                g[0][p] = i as f32;
                g[1][p] = j as f32;
                g[2][p] = k as f32;
                p += 1;
            }
        }
    }
    g.map(|v| Tensor::<3>::from_data(TensorData::new(v, shape), device))
}

/// Rotation matrix entries (row-major, 9 scalars as Tensor<1> of length 1 each) and translation (3).
pub fn pose_matrix(pose: &Tensor<1>) -> ([Tensor<1>; 9], [Tensor<1>; 3]) {
    let r = pose.clone().slice([0..3]) * POSE_ROT;
    let t = pose.clone().slice([3..6]) * POSE_TRANS;
    let (rx, ry, rz) = (r.clone().slice([0..1]), r.clone().slice([1..2]), r.slice([2..3]));
    let (cx, sx) = (rx.clone().cos(), rx.sin());
    let (cy, sy) = (ry.clone().cos(), ry.sin());
    let (cz, sz) = (rz.clone().cos(), rz.sin());
    // R = Rz Ry Rx
    // Rx = [1 0 0; 0 cx -sx; 0 sx cx], Ry = [cy 0 sy; 0 1 0; -sy 0 cy], Rz = [cz -sz 0; sz cz 0; 0 0 1]
    let r00 = cz.clone() * cy.clone();
    let r01 = cz.clone() * sy.clone() * sx.clone() - sz.clone() * cx.clone();
    let r02 = cz.clone() * sy.clone() * cx.clone() + sz.clone() * sx.clone();
    let r10 = sz.clone() * cy.clone();
    let r11 = sz.clone() * sy.clone() * sx.clone() + cz.clone() * cx.clone();
    let r12 = sz * sy.clone() * cx.clone() - cz * sx.clone();
    let r20 = sy.neg();
    let r21 = cy.clone() * sx;
    let r22 = cy * cx;
    ([r00, r01, r02, r10, r11, r12, r20, r21, r22], [t.clone().slice([0..1]), t.clone().slice([1..2]), t.slice([2..3])])
}

fn bc(s: &Tensor<1>, shape: [usize; 3]) -> Tensor<3> {
    s.clone().reshape([1, 1, 1]).expand(shape)
}

/// The reference moved rigidly and resampled onto its own grid.
pub fn posed(reference: Tensor<3>, pose: &Tensor<1>, spacing_mm: [f32; 3], grids: &[Tensor<3>; 3]) -> Tensor<3> {
    let shape = reference.dims();
    let [n0, n1, n2] = shape;
    let device = reference.device();
    let (r, t) = pose_matrix(pose);
    let centre = [0, 1, 2].map(|a| 0.5 * (shape[a] as f32 - 1.0) * spacing_mm[a]);
    // x_mm - centre - t, per axis
    let q: [Tensor<3>; 3] = [0, 1, 2].map(|a| grids[a].clone() * spacing_mm[a] - centre[a] - bc(&t[a], shape));
    // src_mm[k] = sum_j q[j] R[j][k] + centre[k]  (row-vector convention of the prototype)
    let src_idx: [Tensor<3>; 3] = [0, 1, 2].map(|k| {
        let mut s = q[0].clone() * bc(&r[0 * 3 + k], shape);
        s = s + q[1].clone() * bc(&r[1 * 3 + k], shape);
        s = s + q[2].clone() * bc(&r[2 * 3 + k], shape);
        (s + centre[k]) / spacing_mm[k]
    });
    // trilinear gather with zero padding
    let flat = reference.reshape([n0 * n1 * n2]);
    let i0 = [0, 1, 2].map(|a| src_idx[a].clone().floor());
    let fr = [0, 1, 2].map(|a| src_idx[a].clone() - i0[a].clone());
    let mut out: Option<Tensor<3>> = None;
    for corner in 0..8usize {
        let dx = [(corner >> 2) & 1, (corner >> 1) & 1, corner & 1];
        let mut w: Option<Tensor<3>> = None;
        let mut idx: Option<Tensor<3>> = None;
        let mut valid: Option<Tensor<3, Bool>> = None;
        for a in 0..3 {
            let ia = i0[a].clone() + dx[a] as f32;
            let wa = if dx[a] == 1 { fr[a].clone() } else { fr[a].clone().neg() + 1.0 };
            let va = ia.clone().greater_equal_elem(0.0).bool_and(ia.clone().lower_elem(shape[a] as f32));
            w = Some(match w { Some(x) => x * wa, None => wa });
            valid = Some(match valid { Some(x) => x.bool_and(va), None => va });
            let ic = ia.clamp(0.0, shape[a] as f32 - 1.0);
            idx = Some(match idx { Some(x) => x * shape[a] as f32 + ic, None => ic });
        }
        let lin: Tensor<3, Int> = idx.unwrap().int();
        let vals = flat.clone().gather(0, lin.reshape([n0 * n1 * n2])).reshape(shape);
        let term = (vals * w.unwrap()).mask_fill(valid.unwrap().bool_not(), 0.0);
        out = Some(match out { Some(x) => x + term, None => term });
    }
    let _ = device;
    out.unwrap()
}

pub struct PoseReport {
    pub rotation_deg: f32,
    pub translation_mm: [f32; 3],
}

pub fn pose_report(pose: &Tensor<1>) -> PoseReport {
    let (r, t) = pose_matrix(pose);
    let tr = (r[0].clone() + r[4].clone() + r[8].clone()).into_scalar::<f32>();
    let ang = (((tr - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees();
    PoseReport { rotation_deg: ang, translation_mm: [0, 1, 2].map(|a| t[a].clone().into_scalar::<f32>()) }
}
