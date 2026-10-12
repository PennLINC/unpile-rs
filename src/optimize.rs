//! Coarse-to-fine estimation of the field on B-spline control grids with L-BFGS.
use crate::field::{BsplineField, SCALE};
use crate::lbfgs::{lbfgs, LbfgsOptions};
use crate::ops::{bending_energy, from_vec};
use crate::problem::Problem;
use burn::tensor::{Device, Tensor, TensorData};

pub struct OptimizeOptions {
    pub levels_mm: Vec<f32>,
    pub iters: usize,
    pub lam: f32,
    pub rounds: usize,
    /// two-blip mode (problem.obs2 set): latent re-solved between rounds
    pub pepolar: bool,
    /// single-blip joint rigid pose of the reference; the field's in-brain mean is pinned
    /// (a uniform PE shift is a pose, not a field) with `dc_mask`
    pub pose: bool,
    pub dc_mask: Option<Tensor<3>>,
    /// two-blip: block-coordinate L-BFGS on the second blip's pose after each field round
    pub blip_pose: bool,
    /// GRE prior: which parts to refine
    pub gre_refine_pose: bool,
    pub gre_refine_shim: bool,
    pub gre_refine_delta: bool,
}

pub struct LevelReport {
    pub level: usize,
    pub h_mm: f32,
    pub cshape: [usize; 3],
    pub evals: usize,
    pub data0: f32,
    pub data: f32,
    pub reg0: f32,
    pub reg: f32,
    pub gain: f32,
    pub seconds: f64,
    pub pose: Option<[f32; 6]>,
}

/// Returns the full field (Hz) on the image grid (PE-last layout).
pub fn optimize(
    problem: &mut Problem,
    spacing_mm: [f32; 3],
    opt: &OptimizeOptions,
    device: &Device,
    mut log: impl FnMut(&LevelReport),
) -> Tensor<3> {
    let shape = problem.obs.dims();
    let mut f_full: Option<Tensor<3>> = None;
    let mut pose_vec: Vec<f32> = if opt.pose { vec![0.0; 6] } else { vec![] };
    if opt.pose && problem.pose.is_none() {
        problem.enable_pose(spacing_mm);
    }
    let has_gre = problem.gre.is_some();
    // GRE prior parameters live after the pose in the flat vector: 6 pose + 9 shim
    let mut gre_vec: Vec<f32> = if has_gre { vec![0.0; 15] } else { vec![] };
    let lam = if has_gre && !opt.gre_refine_delta {
        1e6
    } else {
        opt.lam
    };
    let nlev = opt.levels_mm.len();
    for (lev, &h_mm) in opt.levels_mm.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let h_vox = [0, 1, 2].map(|a| (h_mm / spacing_mm[a]).round().max(1.0));
        let field = BsplineField::new(shape, h_vox, device);
        let mut c: Vec<f32> = match &f_full {
            Some(ff) => field
                .coefficients_from(ff.clone())
                .to_data()
                .try_to_vec::<f32>()
                .unwrap(),
            None => vec![0.0; field.ncoef()],
        };
        let nc = c.len();
        c.extend_from_slice(&pose_vec); // joint parameter vector: coefficients, then the pose
        let np_ = c.len();
        c.extend_from_slice(&gre_vec);
        let sigma = if lev < nlev - 1 {
            (0.5 * h_mm / spacing_mm[2]).min(3.0)
        } else {
            0.0
        };
        let cshape = field.cshape;
        let mut first: Option<(f32, f32)> = None;
        let mut last: (f32, f32) = (0.0, 0.0);
        let mut evals = 0usize;
        let outside = problem.outside.clone();
        let ow = problem.outside_weight;
        let dc_mask = opt.dc_mask.clone();
        for _round in 0..opt.rounds {
            if opt.pepolar {
                let ct = from_vec(c[..nc].to_vec(), cshape, device);
                let ff = field.full(ct).detach();
                problem.solve_latent(&ff);
            }
            let lbopt = LbfgsOptions {
                max_iter: opt.iters,
                max_eval: 4 * opt.iters,
                ..Default::default()
            };
            let objective = |x: &[f32]| -> (f32, Vec<f32>) {
                let ct = Tensor::<3>::from_data(TensorData::new(x[..nc].to_vec(), cshape), device)
                    .require_grad();
                let pt = if opt.pose {
                    Some(
                        Tensor::<1>::from_data(TensorData::new(x[nc..].to_vec(), [6]), device)
                            .require_grad(),
                    )
                } else {
                    None
                };
                let delta = field.full(ct.clone());
                let (gp, sh) = if has_gre {
                    (
                        Some(
                            Tensor::<1>::from_data(
                                TensorData::new(x[np_..np_ + 6].to_vec(), [6]),
                                device,
                            )
                            .require_grad(),
                        ),
                        Some(
                            Tensor::<1>::from_data(
                                TensorData::new(x[np_ + 6..np_ + 15].to_vec(), [9]),
                                device,
                            )
                            .require_grad(),
                        ),
                    )
                } else {
                    (None, None)
                };
                let f = match problem.prior_field(gp.as_ref(), sh.as_ref()) {
                    Some(pf) => delta.clone() + pf,
                    None => delta.clone(),
                };
                let data = if opt.pepolar {
                    problem.data_term_pepolar(f.clone(), sigma)
                } else {
                    problem.data_term_posed(f.clone(), sigma, pt.as_ref()).0
                };
                let reg = bending_energy(delta.clone(), spacing_mm);
                crate::profile::scope("sync_forward", || {
                    let _ = data.clone().into_scalar::<f32>();
                });
                let mut loss = data.clone() + reg.clone() * lam;
                if has_gre {
                    let dd = delta.clone() / SCALE;
                    loss = loss + (dd.clone() * dd).mean() * problem.gre_ridge;
                }
                if let Some(o) = &outside {
                    let fo = f.clone() / SCALE;
                    loss = loss + (o.clone() * fo.clone() * fo).sum() / o.clone().sum() * ow;
                }
                if let (Some(m), true) = (&dc_mask, opt.pose) {
                    // pin the in-brain mean of the field: confounded with the PE translation of the pose
                    let mean = (f * m.clone()).sum() / m.clone().sum() / SCALE;
                    loss = loss + mean.clone() * mean * 10.0;
                }
                let dv = data.into_scalar::<f32>();
                let rv = reg.into_scalar::<f32>();
                if first.is_none() {
                    first = Some((dv, rv));
                }
                last = (dv, rv);
                evals += 1;
                let grads = crate::profile::scope("backward", || loss.backward());
                let mut g = ct
                    .grad(&grads)
                    .expect("coefficient gradient")
                    .to_data()
                    .try_to_vec::<f32>()
                    .unwrap();
                if let Some(p) = &pt {
                    let gpv = p
                        .grad(&grads)
                        .map(|t| t.to_data().try_to_vec::<f32>().unwrap())
                        .unwrap_or_else(|| vec![0.0; 6]);
                    g.extend_from_slice(&gpv);
                }
                if has_gre {
                    let mut gg = gp
                        .as_ref()
                        .unwrap()
                        .grad(&grads)
                        .map(|t| t.to_data().try_to_vec::<f32>().unwrap())
                        .unwrap_or_else(|| vec![0.0; 6]);
                    if !opt.gre_refine_pose {
                        gg.iter_mut().for_each(|v| *v = 0.0);
                    }
                    let mut gs = sh
                        .as_ref()
                        .unwrap()
                        .grad(&grads)
                        .map(|t| t.to_data().try_to_vec::<f32>().unwrap())
                        .unwrap_or_else(|| vec![0.0; 9]);
                    if !opt.gre_refine_shim {
                        gs.iter_mut().for_each(|v| *v = 0.0);
                    }
                    g.extend_from_slice(&gg);
                    g.extend_from_slice(&gs);
                }
                (loss.into_scalar::<f32>(), g)
            };
            let mut objective = objective;
            lbfgs(&mut objective, &mut c, &lbopt);
            if opt.blip_pose && opt.pepolar {
                // block-coordinate step on the second blip's pose with the field fixed; its PE
                // translation (index 5) is confounded with the field's DC and stays at zero
                let ct = from_vec(c[..nc].to_vec(), cshape, device);
                let f = field.full(ct).detach();
                let mut bp: Vec<f32> = problem
                    .blip_pose
                    .as_ref()
                    .map(|p| p.to_data().try_to_vec::<f32>().unwrap())
                    .unwrap_or(vec![0.0; 6]);
                let bobj = |x: &[f32]| -> (f32, Vec<f32>) {
                    let pt = Tensor::<1>::from_data(TensorData::new(x.to_vec(), [6]), device)
                        .require_grad();
                    let data = problem.data_term_pepolar_posed(f.clone(), sigma, Some(&pt));
                    let grads = data.backward();
                    let mut g = pt
                        .grad(&grads)
                        .map(|t| t.to_data().try_to_vec::<f32>().unwrap())
                        .unwrap_or_else(|| vec![0.0; 6]);
                    g[5] = 0.0;
                    (data.into_scalar::<f32>(), g)
                };
                let mut bobj = bobj;
                lbfgs(
                    &mut bobj,
                    &mut bp,
                    &LbfgsOptions {
                        max_iter: 20,
                        max_eval: 60,
                        history: 10,
                        ..Default::default()
                    },
                );
                problem.blip_pose = Some(Tensor::<1>::from_data(TensorData::new(bp, [6]), device));
            }
        }
        if opt.pose {
            pose_vec = c[nc..np_].to_vec();
            problem.pose = Some(Tensor::<1>::from_data(
                TensorData::new(pose_vec.clone(), [6]),
                device,
            ));
        }
        if has_gre {
            gre_vec = c[np_..].to_vec();
            problem.gre_pose = Some(Tensor::<1>::from_data(
                TensorData::new(gre_vec[..6].to_vec(), [6]),
                device,
            ));
            problem.shim = Some(Tensor::<1>::from_data(
                TensorData::new(gre_vec[6..].to_vec(), [9]),
                device,
            ));
        }
        let ct = from_vec(c[..nc].to_vec(), cshape, device);
        let ff = field.full(ct).detach(); // the residual; the caller adds the prior
        let (d0, r0) = first.unwrap_or((0.0, 0.0));
        let pose_arr = if opt.pose {
            let mut a = [0f32; 6];
            a.copy_from_slice(&pose_vec);
            Some(a)
        } else {
            None
        };
        log(&LevelReport {
            level: lev,
            h_mm,
            cshape,
            evals,
            data0: d0,
            data: last.0,
            reg0: r0,
            reg: last.1,
            gain: problem.last_gain,
            seconds: t0.elapsed().as_secs_f64(),
            pose: pose_arr,
        });
        f_full = Some(ff);
    }
    f_full.unwrap()
}
