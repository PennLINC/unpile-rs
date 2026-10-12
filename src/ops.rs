//! Differentiable building blocks on PE-last tensors [n0, n1, N] (N = phase-encode axis).
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use burn::tensor::module::conv3d;
use burn::tensor::ops::{ConvOptions, IndexingUpdateOp};

const SQRT2: f32 = std::f32::consts::SQRT_2;
const SQRT2PI: f32 = 2.506_628_3;

pub fn from_vec(v: Vec<f32>, shape: [usize; 3], device: &Device) -> Tensor<3> {
    Tensor::<3>::from_data(TensorData::new(v, shape), device)
}

pub fn to_vec(t: &Tensor<3>) -> Vec<f32> {
    t.to_data().try_to_vec::<f32>().expect("float data")
}

/// Integral of the Gaussian CDF: sigma (z Phi(z) + phi(z)).
pub fn psi(x: Tensor<3>, sigma: f32) -> Tensor<3> {
    let z = x / sigma;
    let phi = (z.clone() / SQRT2).erf().add_scalar(1.0) * 0.5;
    let pdf = (z.clone() * z.clone() * -0.5).exp() / SQRT2PI;
    (z * phi + pdf) * sigma
}

fn last(t: &Tensor<3>) -> usize {
    t.dims()[2]
}

/// Central difference along the last axis, one-sided at the ends.
pub fn ddx_last(d: Tensor<3>) -> Tensor<3> {
    let [n0, n1, n] = d.dims();
    let mid = (d.clone().slice([0..n0, 0..n1, 2..n]) - d.clone().slice([0..n0, 0..n1, 0..n - 2])) * 0.5;
    let first = d.clone().slice([0..n0, 0..n1, 1..2]) - d.clone().slice([0..n0, 0..n1, 0..1]);
    let lastd = d.clone().slice([0..n0, 0..n1, n - 1..n]) - d.slice([0..n0, 0..n1, n - 2..n - 1]);
    Tensor::cat(vec![first, mid, lastd], 2)
}

/// Linear interpolation along the last axis at positions `pos`, values fading to zero one voxel
/// beyond either end (continuous in pos everywhere).
pub fn interp_last(img: Tensor<3>, pos: Tensor<3>) -> Tensor<3> {
    crate::profile::scope("interp_last", || interp_last_impl(img, pos))
}
fn interp_last_impl(img: Tensor<3>, pos: Tensor<3>) -> Tensor<3> {
    let [n0, n1, n] = img.dims();
    let device = img.device();
    let zeros = Tensor::<3>::zeros([n0, n1, 1], &device);
    let padded = Tensor::cat(vec![zeros.clone(), img, zeros], 2); // length n + 2
    let p = (pos + 1.0).clamp(0.0, (n + 1) as f32);
    let i0f = p.clone().floor().clamp(0.0, n as f32);
    let w = p - i0f.clone();
    let i0: Tensor<3, Int> = i0f.int();
    let v0 = padded.clone().gather(2, i0.clone());
    let v1 = padded.gather(2, i0 + 1);
    v0 * (w.clone().neg() + 1.0) + v1 * w
}

/// Mass-conserving pushforward along the last axis with Gaussian-smoothed voxel membership.
pub fn splat(img: Tensor<3>, d: Tensor<3>, sigma: f32, nmax: usize) -> Tensor<3> {
    crate::profile::scope("splat", || splat_impl(img, d, sigma, nmax))
}
fn splat_impl(img: Tensor<3>, d: Tensor<3>, sigma: f32, nmax: usize) -> Tensor<3> {
    let [n0, n1, n] = img.dims();
    let device = img.device();
    let de = Tensor::cat(
        vec![
            d.clone().slice([0..n0, 0..n1, 0..1]),
            (d.clone().slice([0..n0, 0..n1, 1..n]) + d.clone().slice([0..n0, 0..n1, 0..n - 1])) * 0.5,
            d.clone().slice([0..n0, 0..n1, n - 1..n]),
        ],
        2,
    ); // [n0, n1, n+1]: displaced half-voxel edges
    let edges = Tensor::<1, Int>::arange(0..(n as i64 + 1), &device).float().sub_scalar(0.5).reshape([1, 1, n + 1]);
    let e = edges.expand([n0, n1, n + 1]) + de;
    let e_lo = e.clone().slice([0..n0, 0..n1, 0..n]);
    let e_hi = e.slice([0..n0, 0..n1, 1..n + 1]);
    let a = e_lo.clone().min_pair(e_hi.clone());
    let b = e_lo.max_pair(e_hi);
    let len = (b.clone() - a.clone()).clamp_min(0.02);
    let j0 = (a.clone() + 0.5).floor() - 1.0;
    let bsz = n0 * n1;
    let flat = |t: Tensor<3>| t.reshape([bsz, n]);
    let (img_f, a_f, b_f, len_f, j0_f) = (flat(img), flat(a), flat(b), flat(len), flat(j0));
    // voxels a stretched cell can reach: its length plus the Gaussian membership tails
    let reach = len_f.clone().max().into_scalar::<f32>().ceil() as usize + 2 + (4.0 * sigma).ceil() as usize;
    let nm = nmax.min(reach).max(1);
    // every offset m at once: [nm, bsz, n] tensors, one scatter, then a sum over m. The cell edges
    // a and b are compared with the nm+1 voxel boundaries j0 + m - 1/2 once each; consecutive
    // boundaries are differenced, so two ramp evaluations serve all nm offsets.
    let lift = |t: Tensor<2>| t.unsqueeze_dim::<3>(0).expand([nm, bsz, n]);
    let lift1 = |t: Tensor<2>| t.unsqueeze_dim::<3>(0).expand([nm + 1, bsz, n]);
    let mvals = Tensor::<1, Int>::arange(0..nm as i64, &device).float().reshape([nm, 1, 1]).expand([nm, bsz, n]);
    let j = lift(j0_f.clone()) + mvals;
    let ov = crate::profile::scope("splat.psi", || {
        let bounds = lift1(j0_f) + Tensor::<1, Int>::arange(0..nm as i64 + 1, &device).float().sub_scalar(0.5).reshape([nm + 1, 1, 1]).expand([nm + 1, bsz, n]);
        let pb = ramp(lift1(b_f) - bounds.clone(), sigma);
        let pa = ramp(lift1(a_f) - bounds, sigma);
        let diff = |p: Tensor<3>| p.clone().slice([0..nm, 0..bsz, 0..n]) - p.slice([1..nm + 1, 0..bsz, 0..n]);
        diff(pb) - diff(pa)
    });
    let (idx, vals) = crate::profile::scope("splat.weights", || {
        let w = ov / lift(len_f);
        let jj: Tensor<3, Int> = j.int();
        let valid: Tensor<3, Bool> = jj.clone().greater_equal_elem(0).bool_and(jj.clone().lower_elem(n as i64));
        let rows = (Tensor::<1, Int>::arange(0..bsz as i64, &device) * (n as i64)).reshape([1, bsz, 1]).expand([nm, bsz, n]);
        let idx = (jj.clamp(0, n as i64 - 1) + rows).reshape([nm, bsz * n]);
        let vals = (lift(img_f) * w).mask_fill(valid.bool_not(), 0.0).reshape([nm, bsz * n]);
        (idx, vals)
    });
    let out = crate::profile::scope("splat.scatter", || {
        Tensor::<2>::zeros([nm, bsz * n], &device).scatter(1, idx, vals, IndexingUpdateOp::Add).sum_dim(0)
    });
    out.reshape([n0, n1, n])
}

/// Which smoothing kernel defines voxel membership in the pushforward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Membership {
    /// Gaussian of the given sigma (voxels): the prototype's kernel, erf/exp per element.
    Gauss,
    /// Triangular kernel of half-width 0.5 voxel (the trilinear hat): its ramp is a clamped
    /// quadratic, so the splat needs no transcendental functions and runs 2-3x faster on the CPU
    /// backend. Equivalent accuracy on every fixture, including folds.
    Tent,
}

static MEMBERSHIP: std::sync::OnceLock<Membership> = std::sync::OnceLock::new();

/// Choose the membership kernel once per process (before the first splat); the default is `Tent`.
pub fn set_membership(m: Membership) {
    let _ = MEMBERSHIP.set(m);
}

pub fn membership() -> Membership {
    *MEMBERSHIP.get_or_init(|| Membership::Tent)
}

/// The smoothed ramp: expected overlap of a half-line with a cell edge smeared by the membership
/// kernel; psi(x) -> x for x >> 0 and -> 0 for x << 0.
fn ramp<const D: usize>(x: Tensor<D>, sigma: f32) -> Tensor<D> {
    match membership() {
        Membership::Tent => {
            let h = 0.5f32;
            let u = x.clone().clamp(-h, h) + h;
            u.clone() * u / (4.0 * h) + (x - h).clamp_min(0.0)
        }
        Membership::Gauss => psi_nd(x, sigma),
    }
}

fn psi_nd<const D: usize>(x: Tensor<D>, sigma: f32) -> Tensor<D> {
    let z = x / sigma;
    let phi = (z.clone() / SQRT2).erf().add_scalar(1.0) * 0.5;
    let pdf = (z.clone() * z.clone() * -0.5).exp() / SQRT2PI;
    (z * phi + pdf) * sigma
}

/// TOPUP-style unwarp of an observation by the Lagrangian displacement d (voxels): sample at
/// y + d(y), scale by the clamped Jacobian 1 + d'(y).
pub fn pullback(img: Tensor<3>, d: Tensor<3>) -> Tensor<3> {
    crate::profile::scope("pullback", || pullback_impl(img, d))
}
fn pullback_impl(img: Tensor<3>, d: Tensor<3>) -> Tensor<3> {
    let [n0, n1, n] = img.dims();
    let device = img.device();
    let y = Tensor::<1, Int>::arange(0..n as i64, &device).float().reshape([1, 1, n]).expand([n0, n1, n]);
    let jac = (ddx_last(d.clone()) + 1.0).clamp_min(0.05);
    interp_last(img, y + d) * jac
}

/// The same resampling without the Jacobian: what `antsApplyTransforms` does with the exported warp.
pub fn pull_lagrangian(img: Tensor<3>, d: Tensor<3>) -> Tensor<3> {
    let [n0, n1, n] = img.dims();
    let device = img.device();
    let y = Tensor::<1, Int>::arange(0..n as i64, &device).float().reshape([1, 1, n]).expand([n0, n1, n]);
    interp_last(img, y + d)
}

fn conv_valid(x: Tensor<3>, kernel: Tensor<5>) -> Tensor<3> {
    let [n0, n1, n2] = x.dims();
    let x5 = x.reshape([1, 1, n0, n1, n2]);
    let y = conv3d(x5, kernel, None, ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1));
    let [_, _, m0, m1, m2] = y.dims();
    y.reshape([m0, m1, m2])
}

/// Box mean in w^3 windows with replicate padding (like avg_pool3d on a replicate-padded input).
pub fn box_mean(x: Tensor<3>, w: usize) -> Tensor<3> {
    crate::profile::scope("box_mean", || box_mean_impl(x, w))
}
fn box_mean_impl(x: Tensor<3>, w: usize) -> Tensor<3> {
    let k = w / 2;
    let device = x.device();
    let mut y = x;
    for ax in 0..3 {
        let mut kshape = [1usize, 1, 1, 1, 1];
        kshape[2 + ax] = w;
        let kernel = Tensor::<5>::ones(kshape, &device) / w as f32;
        y = conv_valid(pad_axis(y, ax, k), kernel);
    }
    y
}

/// Replicate-pad `k` voxels on both sides of one axis.
fn pad_axis(y: Tensor<3>, ax: usize, k: usize) -> Tensor<3> {
    let dims = y.dims();
    let mut r0 = [0..dims[0], 0..dims[1], 0..dims[2]];
    r0[ax] = 0..1;
    let mut r1 = [0..dims[0], 0..dims[1], 0..dims[2]];
    r1[ax] = dims[ax] - 1..dims[ax];
    let first = y.clone().slice(r0);
    let lastv = y.clone().slice(r1);
    let mut parts = Vec::with_capacity(2 * k + 1);
    for _ in 0..k {
        parts.push(first.clone());
    }
    parts.push(y);
    for _ in 0..k {
        parts.push(lastv.clone());
    }
    Tensor::cat(parts, ax)
}

/// 1 - local normalized cross-correlation, averaged over the mask (windows without signal in the
/// observation carry no information).
pub fn lncc_loss(pred: Tensor<3>, obs: Tensor<3>, mask: Tensor<3>, w: usize, eps: f32) -> Tensor<1> {
    crate::profile::scope("lncc_loss", || lncc_loss_impl(pred, obs, mask, w, eps))
}
fn lncc_loss_impl(pred: Tensor<3>, obs: Tensor<3>, mask: Tensor<3>, w: usize, eps: f32) -> Tensor<1> {
    let mp = box_mean(pred.clone(), w);
    let mo = box_mean(obs.clone(), w);
    let vp = box_mean(pred.clone() * pred.clone(), w) - mp.clone() * mp.clone();
    let vo = box_mean(obs.clone() * obs.clone(), w) - mo.clone() * mo.clone();
    let cov = box_mean(pred * obs, w) - mp * mo;
    let cc = cov / ((vp.clamp_min(0.0) * vo.clone().clamp_min(0.0)) + eps).sqrt();
    let valid = mask * vo.greater_elem(eps).float();
    (valid.clone() * (cc.neg() + 1.0)).sum() / valid.sum().clamp_min(1.0)
}

/// Separable Gaussian smoothing (replicate padding), sigma in voxels; identity below 0.3.
pub fn gaussian_smooth(x: Tensor<3>, sigma: f32) -> Tensor<3> {
    crate::profile::scope("gaussian_smooth", || gaussian_smooth_impl(x, sigma))
}
fn gaussian_smooth_impl(x: Tensor<3>, sigma: f32) -> Tensor<3> {
    if sigma <= 0.3 {
        return x;
    }
    let r = (3.0 * sigma).ceil() as i64;
    let device = x.device();
    let kv: Vec<f32> = (-r..=r).map(|i| (-0.5 * (i as f32 / sigma).powi(2)).exp()).collect();
    let s: f32 = kv.iter().sum();
    let kv: Vec<f32> = kv.into_iter().map(|v| v / s).collect();
    let klen = kv.len();
    let mut y = x;
    for ax in 0..3 {
        let mut kshape = [1usize, 1, 1, 1, 1];
        kshape[2 + ax] = klen;
        let kernel = Tensor::<5>::from_data(TensorData::new(kv.clone(), kshape), &device);
        // replicate-pad only along `ax`
        let dims = y.dims();
        let mut r0 = [0..dims[0], 0..dims[1], 0..dims[2]];
        r0[ax] = 0..1;
        let mut r1 = [0..dims[0], 0..dims[1], 0..dims[2]];
        r1[ax] = dims[ax] - 1..dims[ax];
        let first = y.clone().slice(r0);
        let lastv = y.clone().slice(r1);
        let mut parts = Vec::new();
        for _ in 0..r as usize {
            parts.push(first.clone());
        }
        parts.push(y);
        for _ in 0..r as usize {
            parts.push(lastv.clone());
        }
        y = conv_valid(Tensor::cat(parts, ax), kernel);
    }
    y
}

/// Bending energy (sum of squared second derivatives, Hz/mm^2) by circular finite differences,
/// averaged over the volume (matches the prototype, which used torch.roll).
pub fn bending_energy(f: Tensor<3>, spacing_mm: [f32; 3]) -> Tensor<1> {
    crate::profile::scope("bending_energy", || bending_energy_impl(f, spacing_mm))
}
fn bending_energy_impl(f: Tensor<3>, spacing_mm: [f32; 3]) -> Tensor<1> {
    let mut e: Option<Tensor<3>> = None;
    let mut add = |t: Tensor<3>| {
        e = Some(match e.take() {
            Some(acc) => acc + t,
            None => t,
        })
    };
    for a in 0..3 {
        let f2 = (f.clone().roll(&[-1i64], &[a]) - f.clone() * 2.0 + f.clone().roll(&[1i64], &[a])) / spacing_mm[a].powi(2);
        add(f2.clone() * f2);
    }
    for (a, b) in [(0usize, 1usize), (0, 2), (1, 2)] {
        let ga = (f.clone().roll(&[-1i64], &[a]) - f.clone().roll(&[1i64], &[a])) / (2.0 * spacing_mm[a]);
        let gab = (ga.clone().roll(&[-1i64], &[b]) - ga.roll(&[1i64], &[b])) / (2.0 * spacing_mm[b]);
        add(gab.clone() * gab * 2.0);
    }
    e.unwrap().mean()
}

pub fn _n(t: &Tensor<3>) -> usize {
    last(t)
}
