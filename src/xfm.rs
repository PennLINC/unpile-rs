//! The correction as ITK transforms, so `antsApplyTransforms` can apply it.
//!
//! Two files describe a fit: the susceptibility displacement as a 5-D NIfTI vector image
//! (intent 1007, vectors in LPS mm) and, when a reference pose was estimated, an Insight
//! Transform File affine. Both follow ITK's pull convention: a point of the FIXED image
//! (the corrected EPI grid) is mapped to where its value is read in the MOVING image.
//! The RAS/LPS boundary for the affine goes through `itk-transforms-rs`; the warp vectors
//! only need the x/y sign flip of a world vector.
use anyhow::{anyhow, Result};
use ndarray::{Array3, Array5};
use nifti::NiftiHeader;
use std::path::Path;

use crate::pose::{POSE_ROT, POSE_TRANS};

pub type Mat4 = [[f64; 4]; 4];

/// The header's RAS+ voxel-to-world affine (sform when set, else qform: nifti-rs's choice).
pub fn ras_affine(h: &NiftiHeader) -> Mat4 {
    let m = h.affine::<f64>();
    let mut r = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            r[i][j] = m[(i, j)];
        }
    }
    r
}

pub fn matmul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut c = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            c[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

/// Inverse of an affine matrix (last row 0 0 0 1).
pub fn inverse(a: &Mat4) -> Result<Mat4> {
    let l = [
        [a[0][0], a[0][1], a[0][2]],
        [a[1][0], a[1][1], a[1][2]],
        [a[2][0], a[2][1], a[2][2]],
    ];
    let det = l[0][0] * (l[1][1] * l[2][2] - l[1][2] * l[2][1])
        - l[0][1] * (l[1][0] * l[2][2] - l[1][2] * l[2][0])
        + l[0][2] * (l[1][0] * l[2][1] - l[1][1] * l[2][0]);
    if det.abs() < 1e-12 {
        return Err(anyhow!("singular affine"));
    }
    let mut li = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            // cofactor of (j, i)
            let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
            let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
            li[i][j] = (l[r0][c0] * l[r1][c1] - l[r0][c1] * l[r1][c0]) / det;
        }
    }
    let mut out = [[0.0; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = li[i][j];
        }
        out[i][3] = -(li[i][0] * a[0][3] + li[i][1] * a[1][3] + li[i][2] * a[2][3]);
    }
    out[3][3] = 1.0;
    Ok(out)
}

pub fn apply(a: &Mat4, p: [f64; 3]) -> [f64; 3] {
    let mut q = [0.0; 3];
    for i in 0..3 {
        q[i] = a[i][0] * p[0] + a[i][1] * p[1] + a[i][2] * p[2] + a[i][3];
    }
    q
}

/// One voxel step along `pe_axis` in LPS world mm.
pub fn pe_step_lps(h: &NiftiHeader, pe_axis: usize) -> [f64; 3] {
    let a = ras_affine(h);
    [-a[0][pe_axis], -a[1][pe_axis], a[2][pe_axis]]
}

/// Write the field (Hz, in the file's own axis order) as an ITK displacement field on its grid.
///
/// A corrected-grid point is read from the distorted image `sign * f * TRT` voxels further
/// along the PE axis, so the vector is that step in LPS mm: the convention of qsiprep's
/// `_hz_to_warp` and of TOPUP's field.
pub fn write_itk_warp(
    path: impl AsRef<Path>,
    field_hz: &Array3<f32>,
    header: &NiftiHeader,
    pe_axis: usize,
    sign: f32,
    trt: f32,
) -> Result<()> {
    let step = pe_step_lps(header, pe_axis);
    let (nx, ny, nz) = field_hz.dim();
    let mut w = Array5::<f32>::zeros((nx, ny, nz, 1, 3));
    for ((i, j, k), &f) in field_hz.indexed_iter() {
        let s = (sign * f * trt) as f64;
        for c in 0..3 {
            w[[i, j, k, 0, c]] = (s * step[c]) as f32;
        }
    }
    let mut h = header.clone();
    h.intent_code = 1007; // NIFTI_INTENT_VECTOR
    h.intent_p1 = 0.0;
    h.intent_p2 = 0.0;
    h.intent_p3 = 0.0;
    h.intent_name = [0; 16];
    h.cal_min = 0.0;
    h.cal_max = 0.0;
    nifti::writer::WriterOptions::new(path.as_ref())
        .reference_header(&h)
        .write_nifti(&w)?;
    Ok(())
}

/// Rotation matrix of a pose (R = Rz Ry Rx, the prototype's convention), row-major.
fn rotation(pose: &[f32; 6]) -> [[f64; 3]; 3] {
    let (rx, ry, rz) = (
        (pose[0] * POSE_ROT) as f64,
        (pose[1] * POSE_ROT) as f64,
        (pose[2] * POSE_ROT) as f64,
    );
    let (cx, sx, cy, sy, cz, sz) = (rx.cos(), rx.sin(), ry.cos(), ry.sin(), rz.cos(), rz.sin());
    [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
}

/// The pose as a point map in the PE-last grid's mm coordinates (x = index * spacing):
/// `posed()` reads the reference at `R^T (x - c - t) + c`, c the grid centre.
pub fn pose_grid_matrix(pose: &[f32; 6], shape_pl: [usize; 3], spacing_pl: [f32; 3]) -> Mat4 {
    let r = rotation(pose);
    let t: [f64; 3] = [0, 1, 2].map(|a| (pose[3 + a] * POSE_TRANS) as f64);
    let c: [f64; 3] = [0, 1, 2].map(|a| 0.5 * (shape_pl[a] as f64 - 1.0) * spacing_pl[a] as f64);
    let mut m = [[0.0; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = r[j][i]; // R^T
        }
        m[i][3] = c[i] - (0..3).map(|j| r[j][i] * (c[j] + t[j])).sum::<f64>();
    }
    m[3][3] = 1.0;
    m
}

/// PE-last grid mm -> RAS world: A * P * diag(1/spacing), P the PE-last -> file-order permutation.
pub fn grid_to_world(header: &NiftiHeader, pe_axis: usize, spacing_pl: [f32; 3]) -> Mat4 {
    let order: Vec<usize> = (0..3)
        .filter(|&x| x != pe_axis)
        .chain(std::iter::once(pe_axis))
        .collect();
    let mut ps = [[0.0; 4]; 4]; // P * S^-1
    for k in 0..3 {
        ps[order[k]][k] = 1.0 / spacing_pl[k] as f64;
    }
    ps[3][3] = 1.0;
    matmul(&ras_affine(header), &ps)
}

/// The estimated reference pose as a RAS point map, fixed (EPI grid) -> moving (reference).
pub fn pose_world_affine(
    pose: &[f32; 6],
    shape_pl: [usize; 3],
    spacing_pl: [f32; 3],
    pe_axis: usize,
    header: &NiftiHeader,
) -> Result<Mat4> {
    let g = grid_to_world(header, pe_axis, spacing_pl);
    let m = pose_grid_matrix(pose, shape_pl, spacing_pl);
    Ok(matmul(&matmul(&g, &m), &inverse(&g)?))
}

/// Residual `estimate o truth^-1` of two RAS point maps: its rotation angle, and how far apart
/// the maps send `centre` (truth_scoring.rigid_error).
pub fn rigid_error(estimate: &Mat4, truth: &Mat4, centre: [f64; 3]) -> Result<(f64, f64)> {
    let res = matmul(estimate, &inverse(truth)?);
    let tr = res[0][0] + res[1][1] + res[2][2];
    let ang = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
    let (a, b) = (apply(estimate, centre), apply(truth, centre));
    let shift = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    Ok((ang, shift))
}

#[cfg(feature = "itk")]
mod itk {
    use super::*;
    use itk_transforms_rs::{read_itk, Affine3};

    /// Write a RAS point map as an Insight Transform File (LPS, centre 0) via itk-transforms-rs.
    pub fn write_itk_affine_txt(path: impl AsRef<Path>, ras: &Mat4) -> Result<()> {
        let (m, t) = Affine3::from_rows(*ras).into_itk_components();
        let mut s = String::from("#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\nParameters:");
        for i in 0..3 {
            for j in 0..3 {
                s.push_str(&format!(" {:.10}", m[(i, j)]));
            }
        }
        for i in 0..3 {
            s.push_str(&format!(" {:.10}", t[i]));
        }
        s.push_str("\nFixedParameters: 0 0 0\n");
        std::fs::write(path, s)?;
        Ok(())
    }

    /// Read an ITK transform (.txt/.mat/.h5, affine-only) as a RAS point map.
    pub fn read_itk_affine(path: impl AsRef<Path>) -> Result<Mat4> {
        let chain = read_itk(path.as_ref()).map_err(|e| anyhow!("{e}"))?;
        if chain.has_warp() {
            return Err(anyhow!(
                "{:?} contains a displacement field; expected an affine",
                path.as_ref()
            ));
        }
        let o = chain.map_point([0.0, 0.0, 0.0]);
        let mut m = [[0.0; 4]; 4];
        for k in 0..3 {
            let mut e = [0.0; 3];
            e[k] = 1.0;
            let q = chain.map_point(e);
            for i in 0..3 {
                m[i][k] = q[i] - o[i];
            }
        }
        for i in 0..3 {
            m[i][3] = o[i];
        }
        m[3][3] = 1.0;
        Ok(m)
    }
}
#[cfg(feature = "itk")]
pub use itk::{read_itk_affine, write_itk_affine_txt};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_roundtrip() {
        let a = [
            [2.0, 0.1, 0.0, 5.0],
            [0.0, 1.5, 0.2, -3.0],
            [0.3, 0.0, 1.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let i = matmul(&a, &inverse(&a).unwrap());
        for r in 0..4 {
            for c in 0..4 {
                assert!((i[r][c] - if r == c { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn identity_pose_is_identity() {
        let m = pose_grid_matrix(&[0.0; 6], [10, 12, 14], [2.0, 2.0, 2.0]);
        for r in 0..4 {
            for c in 0..4 {
                assert!((m[r][c] - if r == c { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }
}
