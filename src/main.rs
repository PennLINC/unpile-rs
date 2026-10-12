use anyhow::Result;
use burn::tensor::Device;
use clap::{Parser, Subcommand};
use ndarray::Array3;
use serde_json::json;
use unpile::io::{check_same_grid, read3, write3, zooms};
use unpile::layout::{dilate, erode, from_pe_last, median_in, spacing_pe_last, to_pe_last};
use unpile::ops::{from_vec, gaussian_smooth, pullback, splat, to_vec};
use unpile::optimize::{optimize, OptimizeOptions};
use unpile::problem::{Metric, Model, Problem};
use unpile::score::score_field;
use unpile::xfm::{pose_world_affine, rigid_error, write_itk_warp};
#[cfg(feature = "itk")]
use unpile::xfm::{read_itk_affine, write_itk_affine_txt};

#[derive(Parser)]
#[command(name = "unpile", about = "Susceptibility distortion correction with a mass-conserving pushforward model")]
struct Cli {
    /// compute device: flex (pure-Rust CPU) or wgpu (Vulkan/Metal/WebGPU, needs the `wgpu` feature)
    #[arg(long, default_value = "flex", global = true)]
    device: String,
    /// CPU threads for the flex backend (default: all cores; qsiprep passes its per-process allocation)
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// voxel membership kernel of the pushforward: tent (trilinear hat, fast) or gauss (the prototype's)
    #[arg(long, default_value = "tent", global = true)]
    membership: String,
    #[command(subcommand)]
    cmd: Cmd,
}

fn make_device(name: &str) -> Device {
    match name {
        "flex" => Device::flex().autodiff(),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0)).autodiff(),
        // a CPU Vulkan implementation (Mesa lavapipe) when no GPU is present, e.g. in CI
        #[cfg(feature = "wgpu")]
        "wgpu-cpu" => Device::wgpu(burn::tensor::DeviceKind::Cpu).autodiff(),
        #[cfg(feature = "wgpu")]
        "wgpu-default" => Device::wgpu(burn::tensor::DeviceKind::DefaultDevice).autodiff(),
        #[cfg(feature = "cpu")]
        "cpu" => Device::cpu().autodiff(),
        other => panic!("unknown device {other} (built without the wgpu feature?)"),
    }
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Cmd {
    /// Single-blip fit: observed EPI against a reference image.
    Fit {
        #[arg(long)] obs: String,
        /// second blip with the opposite polarity (two-blip mode; --reference is then ignored)
        #[arg(long)] pa: Option<String>,
        #[arg(long)] reference: Option<String>,
        /// estimate a rigid pose of the reference jointly (single blip)
        #[arg(long, default_value_t = false)] pose: bool,
        #[arg(long, default_value_t = false)] blip_pose: bool,
        #[arg(long, default_value_t = false)] blip_gain: bool,
        /// GRE fieldmap (Hz, on the EPI grid, our sign convention) used as a prior
        #[arg(long)] gre: Option<String>,
        #[arg(long, default_value = "pose,shim,delta")] gre_refine: String,
        #[arg(long, default_value_t = 1.0)] gre_ridge: f32,
        /// penalty in displacement units: lam = lam_mm * (TRT * vox_pe)^2 (overrides --lam)
        #[arg(long)] lam_mm: Option<f32>,
        #[arg(long)] mask: String,
        #[arg(long)] out: String,
        #[arg(long)] trt: f32,
        /// PE voxel size of the ACQUISITION (mm) when the images were resampled onto another grid:
        /// Hz x TRT is a shift in acquired voxels, so the shift per grid voxel scales by acq/grid
        #[arg(long)] acq_voxel_mm: Option<f32>,
        #[arg(long, default_value_t = 1)] pe_axis: usize,
        #[arg(long, default_value_t = -1.0)] sign: f32,
        #[arg(long, default_value = "24,12,6,3")] levels: String,
        #[arg(long, default_value_t = 0.1)] lam: f32,
        #[arg(long, default_value_t = 60)] iters: usize,
        #[arg(long, default_value = "ssd")] metric: String,
        #[arg(long, default_value = "push")] model: String,
        #[arg(long, default_value_t = 5)] window: usize,
        #[arg(long)] truth_field: Option<String>,
        /// ITK affine (.txt/.mat/.h5) the reference pose is scored against (fixed EPI -> moving reference)
        #[arg(long)] truth_xfm: Option<String>,
    },
    /// Write a field (and optionally a pose) as ITK transforms, with the internally resampled images for comparison.
    Export {
        /// field in Hz on the EPI grid (the file's own axis order)
        #[arg(long)] field: String,
        /// distorted image to pull back along the field (writes pullback.nii.gz)
        #[arg(long)] obs: Option<String>,
        /// reference image to move by --pose (writes posed.nii.gz)
        #[arg(long)] reference: Option<String>,
        /// six pose parameters as the optimizer reports them (rx,ry,rz,tx,ty,tz in POSE_ROT/POSE_TRANS units)
        #[arg(long)] pose: Option<String>,
        #[arg(long)] out: String,
        #[arg(long)] trt: f32,
        #[arg(long, default_value_t = 1)] pe_axis: usize,
        #[arg(long, default_value_t = -1.0)] sign: f32,
    },
    /// Compare the operators with the PyTorch reference dumps in a testdata directory.
    Check {
        #[arg(long)] dir: String,
    },
}

fn load_mask(path: &str) -> Result<Array3<bool>> {
    Ok(read3(path, None)?.data.map(|&v| v > 0.5))
}

fn stats(name: &str, a: &Array3<f32>, b: &Array3<f32>) {
    let n = a.len() as f64;
    let ma = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mb = b.iter().map(|&v| v as f64).sum::<f64>() / n;
    let (mut sab, mut saa, mut sbb, mut sd) = (0.0, 0.0, 0.0, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64 - ma, y as f64 - mb);
        sab += x * y;
        saa += x * x;
        sbb += y * y;
        sd += (x - y).powi(2);
    }
    let maxd = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    println!("{name:<34}: corr {:.6} rms diff {:.3e} max |diff| {:.3e}", sab / (saa * sbb).sqrt(), (sd / n).sqrt(), maxd);
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?;
    }
    // the stored PyTorch dumps were made with the Gaussian kernel
    let membership = if matches!(cli.cmd, Cmd::Check { .. }) { "gauss" } else { cli.membership.as_str() };
    unpile::ops::set_membership(match membership {
        "tent" => unpile::ops::Membership::Tent,
        "gauss" => unpile::ops::Membership::Gauss,
        other => anyhow::bail!("unknown membership {other}: tent or gauss"),
    });
    if matches!(cli.cmd, Cmd::Check { .. }) {
        println!("check uses the Gaussian membership, the kernel of the stored PyTorch dumps");
    }
    let device = make_device(&cli.device);
    match cli.cmd {
        Cmd::Check { dir } => {
            let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json"))?)?;
            let vals: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/torch_values.json"))?)?;
            let (trt, sign, pe) = (meta["trt"].as_f64().unwrap() as f32, meta["sign"].as_f64().unwrap() as f32, meta["pe_axis"].as_u64().unwrap() as usize);
            let norm = vals["norm"].as_f64().unwrap() as f32;
            let b0 = read3(format!("{dir}/b0.nii.gz"), None)?;
            let clean = read3(format!("{dir}/clean.nii.gz"), None)?.data;
            let ftrue = read3(format!("{dir}/field_true.nii.gz"), None)?.data;
            let brain = load_mask(&format!("{dir}/brain_mask.nii.gz"))?;
            let vox = zooms(&b0.header);
            let sp = spacing_pe_last(vox, pe);
            let t = |a: &Array3<f32>| {
                let p = to_pe_last(a, pe);
                let sh = p.dim();
                from_vec(p.into_raw_vec_and_offset().0, [sh.0, sh.1, sh.2], &device)
            };
            let back = |x: &burn::tensor::Tensor<3>| {
                let [a, b, c] = x.dims();
                from_pe_last(&Array3::from_shape_vec((a, b, c), to_vec(x)).unwrap(), pe)
            };
            let d = t(&ftrue) * (sign * trt);
            let sp_t = back(&splat(t(&(&clean / norm)), d.clone(), 0.5, 10));
            stats("splat(clean) at true field", &sp_t, &read3(format!("{dir}/torch_splat_clean_at_truth.nii.gz"), None)?.data);
            let pb = back(&pullback(t(&(&b0.data / norm)), d));
            stats("pullback(b0) at true field", &pb, &read3(format!("{dir}/torch_pullback_b0_at_truth.nii.gz"), None)?.data);
            let gs = back(&gaussian_smooth(t(&(&b0.data / norm)), 1.5));
            stats("gaussian 1.5 vox", &gs, &read3(format!("{dir}/torch_gauss1p5_b0.nii.gz"), None)?.data);
            let dil = dilate(&brain, vals["mask_dilate_iters"].as_u64().unwrap() as usize).map(|&m| if m { 1.0f32 } else { 0.0 });
            let mut pr = Problem { obs: t(&(&b0.data / norm)), sign, trt, reference: t(&(&clean / norm)), mask_obs: Some(t(&dil)), outside: None, outside_weight: 0.0, model: Model::Push, metric: Metric::Lncc, window: 5, splat_sigma: 0.5, nmax: 10, last_gain: 1.0, obs2: None, pose: None, spacing_mm: sp, grids: None, latent: None, latent_iters: 20, latent_grad_reg: 0.02, blip_pose: None, blip_gain: false, blip_gain_value: 1.0, gre: None, gre_pose: None, shim: None, shim_basis: None, gre_ridge: 1.0 };
            let (l, _) = pr.data_term(t(&ftrue), 0.0);
            println!("{:<34}: rust {:.6}  torch {:.6}", "LNCC at true field", l.into_scalar::<f32>(), vals["lncc_at_truth"]);
            pr.metric = Metric::Ssd;
            let (l, _) = pr.data_term(t(&ftrue), 0.0);
            println!("{:<34}: rust {:.6}  torch {:.6} (gain {:.4} vs {:.4})", "SSD at true field", l.into_scalar::<f32>(), vals["ssd_at_truth"], pr.last_gain, vals["ssd_gain"]);
            let reg = unpile::ops::bending_energy(t(&ftrue), sp).into_scalar::<f32>();
            println!("{:<34}: rust {:.6}  torch {:.6}", "bending energy of true field", reg, vals["bending_at_truth_hz_per_mm2"]);
            // pose resampler
            let pv: Vec<f32> = vals["pose_vector"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let pose = burn::tensor::Tensor::<1>::from_data(burn::tensor::TensorData::new(pv, [6]), &device);
            pr.enable_pose(sp);
            let posed_img = back(&pr.posed_ref(Some(&pose)));
            stats("posed(clean) with a fixed pose", &posed_img, &read3(format!("{dir}/torch_posed_clean.nii.gz"), None)?.data);
            let rep = unpile::pose::pose_report(&pose);
            println!("{:<34}: rust {:.4} deg {:?}  torch {} deg {}", "pose report", rep.rotation_deg, rep.translation_mm, vals["pose_rotation_deg"], vals["pose_translation_mm"]);
            // two-blip latent and cost at the true field
            let pa = read3(format!("{dir}/b0_pa.nii.gz"), None)?.data;
            pr.obs2 = Some(t(&(&pa / norm)));
            pr.mask_obs = None;
            pr.solve_latent(&t(&ftrue));
            let lat = back(pr.latent.as_ref().unwrap());
            stats("latent at true field (CG 20)", &lat, &read3(format!("{dir}/torch_latent_at_truth.nii.gz"), None)?.data);
            let cost = pr.data_term_pepolar(t(&ftrue), 0.0).into_scalar::<f32>();
            let unw = back(&pr.unwarp(pr.obs.clone(), &t(&ftrue), sign, 25, 0.05));
            stats("CG unwarp(b0) at true field", &unw, &read3(format!("{dir}/torch_unwarp_b0_at_truth.nii.gz"), None)?.data);
            println!("{:<34}: rust {:.6}  torch {:.6}", "pepolar cost at true field", cost, vals["pepolar_cost_at_truth"]);
        }
        Cmd::Fit { obs, pa, reference, pose, blip_pose, blip_gain, gre, gre_refine, gre_ridge, lam_mm, mask, out, trt, pe_axis, sign, levels, lam, iters, metric, model, window, truth_field, truth_xfm, acq_voxel_mm } => {
            std::fs::create_dir_all(&out)?;
            let ob = read3(&obs, None)?;
            // every use of trt below is "shift in grid voxels per Hz"; rescale once for a resampled grid
            let trt = match acq_voxel_mm { Some(a) => { let g = zooms(&ob.header)[pe_axis]; println!("  acquisition PE voxel {a} mm on a {g:.3} mm grid: effective TRT {:.5} s", trt * a / g); trt * a / g } None => trt };
            let rf = match &reference { Some(r) => { let v = read3(r, None)?; check_same_grid(&ob, &v, "--reference")?; v.data } None => ob.data.clone() };
            let mk = read3(&mask, None)?;
            check_same_grid(&ob, &mk, "--mask")?;
            let brain = mk.data.map(|&v| v > 0.5);
            if let Some(p) = &pa { check_same_grid(&ob, &read3(p, None)?, "--pa")?; }
            let vox = zooms(&ob.header);
            let sp = spacing_pe_last(vox, pe_axis);
            let norm = median_in(&ob.data, &brain);
            let t = |a: &Array3<f32>| {
                let p = to_pe_last(a, pe_axis);
                let sh = p.dim();
                from_vec(p.into_raw_vec_and_offset().0, [sh.0, sh.1, sh.2], &device)
            };
            let head = {
                let thr = 0.08 * {
                    let mut v: Vec<f32> = ob.data.iter().zip(brain.iter()).filter(|(_, &m)| m).map(|(x, _)| *x).collect();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    v[(0.99 * (v.len() - 1) as f32) as usize]
                };
                let raw = ob.data.map(|&v| v > thr);
                let closed = erode(&dilate(&raw, 2), 2);
                closed.iter().zip(brain.iter()).map(|(&a, &b)| a || b).collect::<Vec<bool>>()
            };
            let head = Array3::from_shape_vec(ob.data.dim(), head)?;
            let outside = dilate(&head, 3).map(|&m| if m { 0.0f32 } else { 1.0 });
            let dil = dilate(&brain, 4).map(|&m| if m { 1.0f32 } else { 0.0 });
            let mut pr = Problem {
                obs: t(&(&ob.data / norm)), sign, trt, reference: t(&(&rf / median_in(&rf, &brain))), mask_obs: Some(t(&dil)), outside: Some(t(&outside)), outside_weight: 1.0,
                model: if model == "pull_lag" { Model::PullLag } else { Model::Push }, metric: if metric == "lncc" { Metric::Lncc } else { Metric::Ssd }, window, splat_sigma: 0.5, nmax: 10, last_gain: 1.0,
                obs2: pa.as_ref().map(|p| t(&(&read3(p, None).unwrap().data / norm))), pose: None, spacing_mm: sp, grids: None, latent: None, latent_iters: 20, latent_grad_reg: 0.02,
                blip_pose: None, blip_gain: false, blip_gain_value: 1.0, gre: None, gre_pose: None, shim: None, shim_basis: None, gre_ridge: 1.0,
            };
            let pepolar = pa.is_some();
            if pepolar {
                pr.mask_obs = None; // the prototype's two-blip data term is unmasked
            }
            let dc_mask = if pose { Some(t(&brain.map(|&m| if m { 1.0f32 } else { 0.0 }))) } else { None };
            if pepolar && (blip_pose || blip_gain) {
                pr.enable_blip(sp, blip_pose, blip_gain);
            }
            if let Some(g) = &gre {
                let gv = read3(g, None)?;
                check_same_grid(&ob, &gv, "--gre")?;
                pr.enable_gre(t(&gv.data), sp);
                pr.gre_ridge = gre_ridge;
            }
            let lam = match lam_mm { Some(l) => { let v = l * (trt * vox[pe_axis]).powi(2); println!("  penalty {l} mm-units -> lam {v:.4}"); v } None => lam };
            let lv: Vec<f32> = levels.split(',').map(|s| s.parse().unwrap()).collect();
            let oo = OptimizeOptions { levels_mm: lv, iters, lam, rounds: 3, pepolar, pose, dc_mask, blip_pose: blip_pose && pepolar, gre_refine_pose: gre_refine.contains("pose"), gre_refine_shim: gre_refine.contains("shim"), gre_refine_delta: gre_refine.contains("delta") };
            let f = optimize(&mut pr, sp, &oo, &device, |r| {
                let ps = match r.pose { Some(p) => { let rep = unpile::pose::pose_report(&burn::tensor::Tensor::<1>::from_data(burn::tensor::TensorData::new(p.to_vec(), [6]), &device)); format!(" pose rot {:.2} deg trans {:+.2},{:+.2},{:+.2} mm", rep.rotation_deg, rep.translation_mm[0], rep.translation_mm[1], rep.translation_mm[2]) } None => String::new() };
                println!("  level {} ({} mm, control {:?}): {} evals, data {:.5} -> {:.5} reg {:.5} -> {:.5} gain {:.3}{} [{:.1} s]", r.level, r.h_mm, r.cshape, r.evals, r.data0, r.data, r.reg0, r.reg, r.gain, ps, r.seconds);
            });
            let f = match pr.prior_field(pr.gre_pose.as_ref(), pr.shim.as_ref()) { Some(pf) => f + pf, None => f };
            if let (Some(gp), Some(sh)) = (&pr.gre_pose, &pr.shim) {
                let rep = unpile::pose::pose_report(gp);
                println!("  GRE refinement: pose rot {:.2} deg trans {:?} mm; shim Hz {:?}", rep.rotation_deg, rep.translation_mm.map(|v| (v * 100.0).round() / 100.0), sh.to_data().try_to_vec::<f32>().unwrap().iter().map(|v| (v * 1000.0).round() / 10.0).collect::<Vec<f32>>());
            }
            if let Some(bp) = &pr.blip_pose {
                let rep = unpile::pose::pose_report(bp);
                println!("  blip: gain {:.3} pose rot {:.2} deg trans {:?} mm", pr.blip_gain_value, rep.rotation_deg, rep.translation_mm.map(|v| (v * 100.0).round() / 100.0));
            }
            let [a, b, c] = f.dims();
            let f_np = from_pe_last(&Array3::from_shape_vec((a, b, c), to_vec(&f))?, pe_axis);
            write3(format!("{out}/field_hz.nii.gz"), &f_np, &ob.header)?;
            write3(format!("{out}/disp_pe_mm.nii.gz"), &f_np.map(|&v| sign * v * trt * vox[pe_axis]), &ob.header)?;
            write_itk_warp(format!("{out}/sdc_warp.nii.gz"), &f_np, &ob.header, pe_axis, sign, trt)?;
            let unw = pr.unwarp(pr.obs.clone(), &f, sign, 25, 0.05);
            let unw_np = from_pe_last(&Array3::from_shape_vec((a, b, c), to_vec(&unw))?, pe_axis).map(|&v| v * norm);
            write3(format!("{out}/corrected.nii.gz"), &unw_np, &ob.header)?;
            if pepolar {
                let unw2 = pr.unwarp(pr.obs2.clone().unwrap(), &f, -sign, 25, 0.05);
                write3(format!("{out}/corrected_pa.nii.gz"), &from_pe_last(&Array3::from_shape_vec((a, b, c), to_vec(&unw2))?, pe_axis).map(|&v| v * norm), &ob.header)?;
            }
            let mut report = json!({"norm": norm, "levels": levels, "lam": lam});
            // the reference pose (fixed EPI -> moving reference) and the GRE prior's pose (fixed EPI -> moving fieldmap)
            for (name, p) in [("pose", pr.pose.as_ref()), ("gre_pose", pr.gre_pose.as_ref())] {
                let Some(p) = p else { continue };
                let pv = p.to_data().try_to_vec::<f32>().unwrap();
                let pa: [f32; 6] = pv[..6].try_into()?;
                let w = pose_world_affine(&pa, [a, b, c], sp, pe_axis, &ob.header)?;
                report[format!("{name}_world_affine_ras")] = json!(w);
                #[cfg(feature = "itk")]
                {
                    write_itk_affine_txt(format!("{out}/{name}_xfm.txt"), &w)?;
                    if let Some(tx) = &truth_xfm {
                        let truth = read_itk_affine(tx)?;
                        let (ang, shift) = rigid_error(&w, &truth, [0.0; 3])?;
                        let (ang0, shift0) = rigid_error(&[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], &truth, [0.0; 3])?;
                        println!("SCORE {name} vs truth: residual rotation {ang:.2} deg, translation {shift:.2} mm at the origin (identity would leave {ang0:.2} deg, {shift0:.2} mm)");
                        report[format!("{name}_vs_truth")] = json!({"rotation_deg": ang, "translation_mm": shift, "identity_rotation_deg": ang0, "identity_translation_mm": shift0});
                    }
                }
            }
            if let Some(tf) = truth_field {
                let ftv = read3(&tf, None)?;
                check_same_grid(&ob, &ftv, "--truth-field")?;
                let ft = ftv.data;
                let er = erode(&brain, 2);
                let s = score_field(&f_np.map(|&v| sign * v * trt), &ft.map(|&v| sign * v * trt), &er, vox[pe_axis]);
                if let Some(p) = &pr.pose {
                    let rep = unpile::pose::pose_report(p);
                    let dc = rep.translation_mm[2] / (sign * trt * vox[pe_axis]);
                    let s2 = score_field(&f_np.map(|&v| sign * (v + dc) * trt), &ft.map(|&v| sign * v * trt), &er, vox[pe_axis]);
                    println!("  DC from pose: {:+.1} Hz -> un-demeaned rms {:.2} mm (pinned field {:.2})", dc, s2.rms_residual_mm, s.rms_residual_mm);
                }
                println!("SCORE field corr {:.3} slope {:.3} rms {:.2} of {:.2} mm (demeaned: slope {:.3} rms {:.2}; mean est/true {:.2}/{:.2} mm)", s.corr, s.slope, s.rms_residual_mm, s.rms_truth_mm, s.slope_demeaned, s.rms_residual_demeaned_mm, s.mean_est_mm, s.mean_truth_mm);
                report["field"] = serde_json::to_value(&s)?;
            }
            std::fs::write(format!("{out}/score.json"), serde_json::to_string_pretty(&report)?)?;
        }
        Cmd::Export { field, obs, reference, pose, out, trt, pe_axis, sign } => {
            std::fs::create_dir_all(&out)?;
            let fld = read3(&field, None)?;
            let vox = zooms(&fld.header);
            let sp = spacing_pe_last(vox, pe_axis);
            let t = |a: &Array3<f32>| {
                let p = to_pe_last(a, pe_axis);
                let sh = p.dim();
                from_vec(p.into_raw_vec_and_offset().0, [sh.0, sh.1, sh.2], &device)
            };
            let back = |x: &burn::tensor::Tensor<3>| {
                let [a, b, c] = x.dims();
                from_pe_last(&Array3::from_shape_vec((a, b, c), to_vec(x)).unwrap(), pe_axis)
            };
            write_itk_warp(format!("{out}/sdc_warp.nii.gz"), &fld.data, &fld.header, pe_axis, sign, trt)?;
            println!("wrote {out}/sdc_warp.nii.gz (PE step LPS mm {:?})", unpile::xfm::pe_step_lps(&fld.header, pe_axis));
            if let Some(o) = obs {
                let ob = read3(&o, None)?;
                check_same_grid(&fld, &ob, "--obs")?;
                let d = t(&fld.data) * (sign * trt);
                write3(format!("{out}/pullback.nii.gz"), &back(&pullback(t(&ob.data), d.clone())), &ob.header)?;
                write3(format!("{out}/pull_nojac.nii.gz"), &back(&unpile::ops::pull_lagrangian(t(&ob.data), d)), &ob.header)?;
            }
            if let Some(ps) = pose {
                let pv: Vec<f32> = ps.split(',').map(|v| v.trim().parse::<f32>()).collect::<std::result::Result<_, _>>()?;
                let pa: [f32; 6] = pv.clone().try_into().map_err(|_| anyhow::anyhow!("--pose needs six values"))?;
                let shp = t(&fld.data).dims();
                let w = pose_world_affine(&pa, shp, sp, pe_axis, &fld.header)?;
                println!("pose world affine (RAS, fixed EPI -> moving reference):");
                for r in &w {
                    println!("  {:+.6} {:+.6} {:+.6} {:+.4}", r[0], r[1], r[2], r[3]);
                }
                #[cfg(feature = "itk")]
                write_itk_affine_txt(format!("{out}/pose_xfm.txt"), &w)?;
                if let Some(r) = reference {
                    let rf = read3(&r, None)?;
                    check_same_grid(&fld, &rf, "--reference")?;
                    let grids = unpile::pose::index_grids(shp, &device);
                    let pt = burn::tensor::Tensor::<1>::from_data(burn::tensor::TensorData::new(pv, [6]), &device);
                    write3(format!("{out}/posed.nii.gz"), &back(&unpile::pose::posed(t(&rf.data), &pt, sp, &grids)), &rf.header)?;
                }
            }
        }
    }
    unpile::profile::report();
    Ok(())
}
