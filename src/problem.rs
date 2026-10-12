//! The single-blip estimation problem: an observed (distorted) EPI, a reference of the same
//! contrast (clean b=0, SynB0) or another contrast (T2w with LNCC), and the forward model.
use crate::latent::{adjoint, cg_solve};
use crate::ops::{gaussian_smooth, lncc_loss, pullback, splat};
use crate::pose::{index_grids, posed};
use burn::tensor::Tensor;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Model {
    Push,
    PullLag,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Metric {
    Ssd,
    Lncc,
}

pub struct Problem {
    pub obs: Tensor<3>,
    /// second blip (two-blip mode), with the opposite sign
    pub obs2: Option<Tensor<3>>,
    pub sign: f32,
    pub trt: f32,
    pub reference: Tensor<3>,
    pub mask_obs: Option<Tensor<3>>,
    pub outside: Option<Tensor<3>>,
    pub outside_weight: f32,
    pub model: Model,
    pub metric: Metric,
    pub window: usize,
    pub splat_sigma: f32,
    pub nmax: usize,
    pub last_gain: f32,
    /// rigid pose of the reference (6 parameters, prototype units), single-blip mode
    pub pose: Option<Tensor<1>>,
    pub spacing_mm: [f32; 3],
    pub grids: Option<[Tensor<3>; 3]>,
    /// two-blip latent image, held fixed within an L-BFGS round
    pub latent: Option<Tensor<3>>,
    pub latent_iters: usize,
    pub latent_grad_reg: f32,
    /// two-blip: rigid pose of the second blip relative to the latent (PE translation held at 0)
    pub blip_pose: Option<Tensor<1>>,
    /// two-blip: closed-form intensity gain of the second blip
    pub blip_gain: bool,
    pub blip_gain_value: f32,
    /// GRE prior (Hz on the data grid) with its own pose and a harmonic shim polynomial
    pub gre: Option<Tensor<3>>,
    pub gre_pose: Option<Tensor<1>>,
    pub shim: Option<Tensor<1>>,
    pub shim_basis: Option<Tensor<2>>, // (9, n0*n1*n2)
    pub gre_ridge: f32,
}

/// Constant + eight solid harmonics up to second order in physical (mm) coordinates about the
/// grid centre, one common scale (largest half-extent): what shim coils can produce.
pub fn harmonic_basis(shape: [usize; 3], spacing_mm: [f32; 3], device: &burn::tensor::Device) -> Tensor<2> {
    let [n0, n1, n2] = shape;
    let half = [0, 1, 2].map(|a| (shape[a] as f32 - 1.0) / 2.0 * spacing_mm[a]);
    let scale = half.iter().cloned().fold(0.0f32, f32::max);
    let nvox = n0 * n1 * n2;
    let mut v = vec![0f32; 9 * nvox];
    let mut p = 0;
    for i in 0..n0 {
        for j in 0..n1 {
            for k in 0..n2 {
                let u = (i as f32 * spacing_mm[0] - half[0]) / scale;
                let w1 = (j as f32 * spacing_mm[1] - half[1]) / scale;
                let w = (k as f32 * spacing_mm[2] - half[2]) / scale;
                let m = [1.0, u, w1, w, u * w1, u * w, w1 * w, u * u - w1 * w1, 2.0 * w * w - u * u - w1 * w1];
                for (b, val) in m.iter().enumerate() {
                    v[b * nvox + p] = *val;
                }
                p += 1;
            }
        }
    }
    Tensor::<2>::from_data(burn::tensor::TensorData::new(v, [9, nvox]), device)
}

impl Problem {
    pub fn disp(&self, f: Tensor<3>) -> Tensor<3> {
        f * (self.sign * self.trt)
    }

    pub fn enable_blip(&mut self, spacing_mm: [f32; 3], pose: bool, gain: bool) {
        self.spacing_mm = spacing_mm;
        if self.grids.is_none() {
            self.grids = Some(index_grids(self.obs.dims(), &self.obs.device()));
        }
        if pose {
            self.blip_pose = Some(Tensor::<1>::zeros([6], &self.obs.device()));
        }
        self.blip_gain = gain;
    }

    pub fn enable_gre(&mut self, gre_hz: Tensor<3>, spacing_mm: [f32; 3]) {
        self.spacing_mm = spacing_mm;
        if self.grids.is_none() {
            self.grids = Some(index_grids(self.obs.dims(), &self.obs.device()));
        }
        let device = gre_hz.device();
        self.shim_basis = Some(harmonic_basis(gre_hz.dims(), spacing_mm, &device));
        self.gre = Some(gre_hz);
        self.gre_pose = Some(Tensor::<1>::zeros([6], &device));
        self.shim = Some(Tensor::<1>::zeros([9], &device));
    }

    /// The GRE field under its pose plus the shim polynomial (Hz); zero without a prior.
    pub fn prior_field(&self, gre_pose: Option<&Tensor<1>>, shim: Option<&Tensor<1>>) -> Option<Tensor<3>> {
        let gre = self.gre.as_ref()?;
        let shape = gre.dims();
        let g = match (gre_pose, &self.grids) {
            (Some(p), Some(gr)) => posed(gre.clone(), p, self.spacing_mm, gr),
            _ => gre.clone(),
        };
        let poly = match (shim, &self.shim_basis) {
            (Some(c), Some(b)) => c.clone().reshape([1, 9]).matmul(b.clone()).reshape(shape) * 100.0,
            _ => Tensor::zeros(shape, &gre.device()),
        };
        Some(g + poly)
    }

    pub fn enable_pose(&mut self, spacing_mm: [f32; 3]) {
        self.spacing_mm = spacing_mm;
        self.grids = Some(index_grids(self.reference.dims(), &self.reference.device()));
        self.pose = Some(Tensor::<1>::zeros([6], &self.reference.device()));
    }

    pub fn posed_ref(&self, pose: Option<&Tensor<1>>) -> Tensor<3> {
        crate::profile::scope("posed_ref", || self.posed_ref_impl(pose))
    }
    fn posed_ref_impl(&self, pose: Option<&Tensor<1>>) -> Tensor<3> {
        match (pose, &self.grids) {
            (Some(p), Some(g)) => posed(self.reference.clone(), p, self.spacing_mm, g),
            _ => self.reference.clone(),
        }
    }

    fn forward(&self, img: Tensor<3>, f: Tensor<3>, sign: f32) -> Tensor<3> {
        splat(img, f * (sign * self.trt), self.splat_sigma, self.nmax)
    }

    /// Regularised inverse of the pushforward for one observation (the corrected image).
    pub fn unwarp(&self, obs: Tensor<3>, f: &Tensor<3>, sign: f32, n_iter: usize, grad_reg: f32) -> Tensor<3> {
        let fd = f.clone().detach();
        let a = |x: Tensor<3>, _k: usize| self.forward(x, fd.clone(), sign);
        let at = |r: Tensor<3>, _k: usize| adjoint(|z| self.forward(z, fd.clone(), sign), &obs, r);
        cg_solve(a, at, std::slice::from_ref(&obs), obs.clone(), n_iter, 1e-3, grad_reg).detach()
    }

    /// Two-blip: re-solve the latent image for the current field (no gradient).
    pub fn solve_latent(&mut self, f: &Tensor<3>) {
        crate::profile::scope("solve_latent", || self.solve_latent_impl(f))
    }
    fn solve_latent_impl(&mut self, f: &Tensor<3>) {
        let obs2 = self.obs2.clone().expect("two-blip mode");
        let fd = f.clone().detach();
        let signs = [self.sign, -self.sign];
        let x0 = self.latent.clone().unwrap_or_else(|| (self.obs.clone() + obs2.clone()) * 0.5);
        let bp = self.blip_pose.clone().map(|p| p.detach());
        let g1 = if self.blip_gain { self.blip_gain_value } else { 1.0 };
        let grids = self.grids.clone();
        let sp = self.spacing_mm;
        let a = |x: Tensor<3>, k: usize| {
            if k == 1 {
                let src = match (&bp, &grids) { (Some(p), Some(g)) => posed(x, p, sp, g), _ => x };
                self.forward(src, fd.clone(), signs[k]) * g1
            } else {
                self.forward(x, fd.clone(), signs[k])
            }
        };
        let at = |r: Tensor<3>, k: usize| adjoint(|z| a(z, k), &x0, r);
        let rhs = [self.obs.clone(), obs2];
        self.latent = Some(cg_solve(a, at, &rhs, x0.clone(), self.latent_iters, 1e-3, self.latent_grad_reg).detach());
    }

    /// Two-blip data term with the latent held fixed.
    pub fn data_term_pepolar(&mut self, f: Tensor<3>, smooth_sigma: f32) -> Tensor<1> {
        crate::profile::scope("data_term_pepolar", || self.data_term_pepolar_impl(f, smooth_sigma))
    }
    fn data_term_pepolar_impl(&mut self, f: Tensor<3>, smooth_sigma: f32) -> Tensor<1> {
        let bp = self.blip_pose.clone();
        self.data_term_pepolar_posed(f, smooth_sigma, bp.as_ref())
    }

    pub fn data_term_pepolar_posed(&mut self, f: Tensor<3>, smooth_sigma: f32, blip_pose: Option<&Tensor<1>>) -> Tensor<1> {
        if self.latent.is_none() {
            self.solve_latent(&f);
        }
        let lat = self.latent.clone().unwrap();
        let obs2 = self.obs2.clone().expect("two-blip mode");
        let signs = [self.sign, -self.sign];
        let mut cost: Option<Tensor<1>> = None;
        for (k, o) in [self.obs.clone(), obs2].into_iter().enumerate() {
            let src = if k == 1 { match (blip_pose, &self.grids) { (Some(p), Some(g)) => posed(lat.clone(), p, self.spacing_mm, g), _ => lat.clone() } } else { lat.clone() };
            let mut p = self.forward(src, f.clone(), signs[k]);
            if k == 1 && self.blip_gain {
                let m = self.mask_obs.clone().unwrap_or_else(|| Tensor::ones(o.dims(), &o.device()));
                let g = (m.clone() * p.clone() * o.clone()).sum() / (m * p.clone() * p.clone()).sum().clamp_min(1e-12);
                self.blip_gain_value = g.clone().into_scalar::<f32>();
                p = p * g.unsqueeze::<3>().expand(o.dims());
            }
            let mut o = o;
            if smooth_sigma > 0.0 {
                p = gaussian_smooth(p, smooth_sigma);
                o = gaussian_smooth(o, smooth_sigma);
            }
            let diff = p - o;
            let term = match &self.mask_obs {
                Some(m) => (m.clone() * diff.clone() * diff).sum() / m.clone().sum().clamp_min(1.0),
                None => (diff.clone() * diff).mean(),
            };
            cost = Some(match cost { Some(c) => c + term, None => term });
        }
        cost.unwrap()
    }

    /// Data term and the prediction it was computed from.
    pub fn data_term(&mut self, f: Tensor<3>, smooth_sigma: f32) -> (Tensor<1>, Tensor<3>) {
        self.data_term_posed(f, smooth_sigma, None)
    }

    pub fn data_term_posed(&mut self, f: Tensor<3>, smooth_sigma: f32, pose: Option<&Tensor<1>>) -> (Tensor<1>, Tensor<3>) {
        crate::profile::scope("data_term_posed", || self.data_term_posed_impl(f, smooth_sigma, pose))
    }
    fn data_term_posed_impl(&mut self, f: Tensor<3>, smooth_sigma: f32, pose: Option<&Tensor<1>>) -> (Tensor<1>, Tensor<3>) {
        let d = self.disp(f);
        let reference = self.posed_ref(pose);
        let (mut pred, mut obs) = match self.model {
            Model::Push => (splat(reference, d, self.splat_sigma, self.nmax), self.obs.clone()),
            // TOPUP-style control: compare the unwarped observation with the reference
            Model::PullLag => (reference, pullback(self.obs.clone(), d)),
        };
        if smooth_sigma > 0.0 {
            pred = gaussian_smooth(pred, smooth_sigma);
            obs = gaussian_smooth(obs, smooth_sigma);
        }
        let m = self.mask_obs.clone().unwrap_or_else(|| Tensor::ones(obs.dims(), &obs.device()));
        match self.metric {
            Metric::Lncc => {
                let l = lncc_loss(pred.clone(), obs, m, self.window, 1e-6);
                (l, pred)
            }
            Metric::Ssd => {
                // global intensity gain in closed form (least squares inside the mask)
                let num = (m.clone() * pred.clone() * obs.clone()).sum();
                let den = (m.clone() * pred.clone() * pred.clone()).sum().clamp_min(1e-12);
                let gain = (num / den).detach();
                self.last_gain = gain.clone().into_scalar::<f32>();
                let p = pred * gain.unsqueeze::<3>().expand(obs.dims());
                let diff = p.clone() - obs;
                let l = (m.clone() * diff.clone() * diff).sum() / m.sum();
                (l, p)
            }
        }
    }
}
