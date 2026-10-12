//! Cubic B-spline control-grid field (C2), coefficients in units of SCALE Hz.
use burn::tensor::{Device, Tensor, TensorData};

pub const SCALE: f32 = 100.0;

pub struct BsplineField {
    pub shape: [usize; 3],
    pub cshape: [usize; 3],
    m: [Tensor<2>; 3],    // (n_i x K_i)
    pinv: [Tensor<2>; 3], // (K_i x n_i)
}

fn bspline_matrix(n: usize, h: f32) -> (Vec<f64>, usize) {
    let k_in = ((n as f64 - 1.0) / h as f64).ceil() as usize + 1;
    let kk = k_in + 2;
    let mut m = vec![0.0f64; n * kk];
    for x in 0..n {
        let u = x as f64 / h as f64;
        let i = u.floor() as i64;
        let t = u - i as f64;
        let w = [
            (1.0 - t).powi(3) / 6.0,
            (3.0 * t.powi(3) - 6.0 * t.powi(2) + 4.0) / 6.0,
            (-3.0 * t.powi(3) + 3.0 * t.powi(2) + 3.0 * t + 1.0) / 6.0,
            t.powi(3) / 6.0,
        ];
        for j in 0..4 {
            let col = (i + j as i64).clamp(0, kk as i64 - 1) as usize;
            m[x * kk + col] += w[j];
        }
    }
    (m, kk)
}

/// Pseudo-inverse of an (n x k) matrix via the ridge-regularised normal equations, Gauss-Jordan.
fn pinv(m: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for r in 0..n {
                s += m[r * k + i] * m[r * k + j];
            }
            a[i * k + j] = s + if i == j { 1e-9 } else { 0.0 };
        }
    }
    // invert a (k x k)
    let mut inv = vec![0.0f64; k * k];
    for i in 0..k {
        inv[i * k + i] = 1.0;
    }
    for c in 0..k {
        let mut p = c;
        for r in c + 1..k {
            if a[r * k + c].abs() > a[p * k + c].abs() {
                p = r;
            }
        }
        if p != c {
            for j in 0..k {
                a.swap(c * k + j, p * k + j);
                inv.swap(c * k + j, p * k + j);
            }
        }
        let d = a[c * k + c];
        for j in 0..k {
            a[c * k + j] /= d;
            inv[c * k + j] /= d;
        }
        for r in 0..k {
            if r != c {
                let f = a[r * k + c];
                if f != 0.0 {
                    for j in 0..k {
                        a[r * k + j] -= f * a[c * k + j];
                        inv[r * k + j] -= f * inv[c * k + j];
                    }
                }
            }
        }
    }
    // pinv = inv (k x k) * M^T (k x n)
    let mut out = vec![0.0f64; k * n];
    for i in 0..k {
        for r in 0..n {
            let mut s = 0.0;
            for j in 0..k {
                s += inv[i * k + j] * m[r * k + j];
            }
            out[i * n + r] = s;
        }
    }
    out
}

/// out[i,j,k] = sum_abc A0[i,a] A1[j,b] A2[k,c] x[a,b,c]
pub fn apply3(x: Tensor<3>, a0: &Tensor<2>, a1: &Tensor<2>, a2: &Tensor<2>) -> Tensor<3> {
    let [s0, s1, s2] = x.dims();
    let m0 = a0.dims()[0];
    let m1 = a1.dims()[0];
    let m2 = a2.dims()[0];
    let t = x.reshape([s0 * s1, s2]).matmul(a2.clone().transpose()).reshape([s0, s1, m2]); // (s0,s1,m2)
    let t = t.swap_dims(1, 2).reshape([s0 * m2, s1]).matmul(a1.clone().transpose()).reshape([s0, m2, m1]); // (s0,m2,m1)
    let t = t.permute([1, 2, 0]).reshape([m2 * m1, s0]).matmul(a0.clone().transpose()).reshape([m2, m1, m0]); // (m2,m1,m0)
    t.permute([2, 1, 0])
}

impl BsplineField {
    pub fn new(shape: [usize; 3], spacing_vox: [f32; 3], device: &Device) -> Self {
        let mut m = Vec::new();
        let mut pv = Vec::new();
        let mut cshape = [0usize; 3];
        for ax in 0..3 {
            let (mat, kk) = bspline_matrix(shape[ax], spacing_vox[ax]);
            cshape[ax] = kk;
            let p = pinv(&mat, shape[ax], kk);
            m.push(Tensor::<2>::from_data(TensorData::new(mat.iter().map(|&v| v as f32).collect::<Vec<f32>>(), [shape[ax], kk]), device));
            pv.push(Tensor::<2>::from_data(TensorData::new(p.iter().map(|&v| v as f32).collect::<Vec<f32>>(), [kk, shape[ax]]), device));
        }
        let m: [Tensor<2>; 3] = [m.remove(0), m.remove(0), m.remove(0)];
        let pinv: [Tensor<2>; 3] = [pv.remove(0), pv.remove(0), pv.remove(0)];
        Self { shape, cshape, m, pinv }
    }

    /// Field (Hz) on the image grid from coefficients (units of SCALE Hz).
    pub fn full(&self, c: Tensor<3>) -> Tensor<3> {
        crate::profile::scope("full", || self.full_impl(c))
    }
    fn full_impl(&self, c: Tensor<3>) -> Tensor<3> {
        apply3(c, &self.m[0], &self.m[1], &self.m[2]) * SCALE
    }

    /// Least-squares coefficients reproducing a full field (Hz).
    pub fn coefficients_from(&self, full: Tensor<3>) -> Tensor<3> {
        apply3(full / SCALE, &self.pinv[0], &self.pinv[1], &self.pinv[2])
    }

    pub fn ncoef(&self) -> usize {
        self.cshape.iter().product()
    }
}
