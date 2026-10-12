//! L-BFGS with a strong-Wolfe line search (cubic interpolation), after torch.optim.LBFGS.
use std::collections::VecDeque;

pub struct LbfgsOptions {
    pub max_iter: usize,
    pub max_eval: usize,
    pub history: usize,
    pub tolerance_grad: f32,
    pub tolerance_change: f32,
    pub lr: f32,
}

impl Default for LbfgsOptions {
    fn default() -> Self {
        Self {
            max_iter: 60,
            max_eval: 240,
            history: 20,
            tolerance_grad: 1e-10,
            tolerance_change: 1e-14,
            lr: 1.0,
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

fn cubic_interpolate(
    x1: f64,
    f1: f64,
    g1: f64,
    x2: f64,
    f2: f64,
    g2: f64,
    bounds: Option<(f64, f64)>,
) -> f64 {
    let (xmin, xmax) = bounds.unwrap_or(if x1 <= x2 { (x1, x2) } else { (x2, x1) });
    let d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2);
    let d2_square = d1 * d1 - g1 * g2;
    if d2_square >= 0.0 {
        let d2 = d2_square.sqrt();
        let min_pos = if x1 <= x2 {
            x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2.0 * d2))
        } else {
            x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2.0 * d2))
        };
        min_pos.max(xmin).min(xmax)
    } else {
        (xmin + xmax) / 2.0
    }
}

/// Returns (f_new, g_new, t, n_evals). `obj(x)` -> (f, grad).
fn strong_wolfe<F: FnMut(&[f32]) -> (f32, Vec<f32>)>(
    obj: &mut F,
    x: &[f32],
    t0: f64,
    d: &[f32],
    f: f64,
    g: &[f32],
    gtd: f64,
    c1: f64,
    c2: f64,
    tol_change: f64,
    max_ls: usize,
) -> (f64, Vec<f32>, f64, usize) {
    let eval_at = |obj: &mut F, t: f64| -> (f64, Vec<f32>) {
        let xn: Vec<f32> = x
            .iter()
            .zip(d)
            .map(|(xi, di)| xi + (t as f32) * di)
            .collect();
        let (fv, gv) = obj(&xn);
        (fv as f64, gv)
    };
    let d_norm = d.iter().fold(0.0f64, |m, v| m.max(v.abs() as f64));
    let mut t = t0;
    let (mut f_new, mut g_new) = eval_at(obj, t);
    let mut ls_evals = 1usize;
    let mut gtd_new = dot(&g_new, d);
    let (mut t_prev, mut f_prev, mut g_prev, mut gtd_prev) = (0.0f64, f, g.to_vec(), gtd);
    let mut done = false;
    let mut ls_iter = 0usize;
    let mut bracket: Vec<f64> = vec![];
    let mut bracket_f: Vec<f64> = vec![];
    let mut bracket_g: Vec<Vec<f32>> = vec![];
    let mut bracket_gtd: Vec<f64> = vec![];
    while ls_iter < max_ls {
        if f_new > f + c1 * t * gtd || (ls_iter > 1 && f_new >= f_prev) {
            bracket = vec![t_prev, t];
            bracket_f = vec![f_prev, f_new];
            bracket_g = vec![g_prev.clone(), g_new.clone()];
            bracket_gtd = vec![gtd_prev, gtd_new];
            break;
        }
        if gtd_new.abs() <= -c2 * gtd {
            bracket = vec![t];
            bracket_f = vec![f_new];
            bracket_g = vec![g_new.clone()];
            done = true;
            break;
        }
        if gtd_new >= 0.0 {
            bracket = vec![t_prev, t];
            bracket_f = vec![f_prev, f_new];
            bracket_g = vec![g_prev.clone(), g_new.clone()];
            bracket_gtd = vec![gtd_prev, gtd_new];
            break;
        }
        let min_step = t + 0.01 * (t - t_prev);
        let max_step = t * 10.0;
        let tmp = t;
        t = cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            Some((min_step, max_step)),
        );
        t_prev = tmp;
        f_prev = f_new;
        g_prev = g_new.clone();
        gtd_prev = gtd_new;
        let r = eval_at(obj, t);
        f_new = r.0;
        g_new = r.1;
        ls_evals += 1;
        gtd_new = dot(&g_new, d);
        ls_iter += 1;
    }
    if ls_iter == max_ls {
        bracket = vec![0.0, t];
        bracket_f = vec![f, f_new];
        bracket_g = vec![g.to_vec(), g_new.clone()];
        bracket_gtd = vec![gtd, gtd_new];
    }
    // zoom
    let mut insuf_progress = false;
    let (mut low_pos, mut high_pos) = if bracket_f.len() == 2 && bracket_f[0] <= bracket_f[1] {
        (0usize, 1usize)
    } else {
        (1usize, 0usize)
    };
    while !done && ls_iter < max_ls {
        if bracket.len() < 2 {
            break;
        }
        if (bracket[1] - bracket[0]).abs() * d_norm < tol_change {
            break;
        }
        t = cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
            None,
        );
        let bmax = bracket[0].max(bracket[1]);
        let bmin = bracket[0].min(bracket[1]);
        let eps = 0.1 * (bmax - bmin);
        if (bmax - t).min(t - bmin) < eps {
            if insuf_progress || t >= bmax || t <= bmin {
                t = if (t - bmax).abs() < (t - bmin).abs() {
                    bmax - eps
                } else {
                    bmin + eps
                };
                insuf_progress = false;
            } else {
                insuf_progress = true;
            }
        } else {
            insuf_progress = false;
        }
        let r = eval_at(obj, t);
        f_new = r.0;
        g_new = r.1;
        ls_evals += 1;
        gtd_new = dot(&g_new, d);
        ls_iter += 1;
        if f_new > f + c1 * t * gtd || f_new >= bracket_f[low_pos] {
            bracket[high_pos] = t;
            bracket_f[high_pos] = f_new;
            bracket_g[high_pos] = g_new.clone();
            bracket_gtd[high_pos] = gtd_new;
            let (l, h) = if bracket_f[0] <= bracket_f[1] {
                (0, 1)
            } else {
                (1, 0)
            };
            low_pos = l;
            high_pos = h;
        } else {
            if gtd_new.abs() <= -c2 * gtd {
                done = true;
            } else if gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0.0 {
                bracket[high_pos] = bracket[low_pos];
                bracket_f[high_pos] = bracket_f[low_pos];
                bracket_g[high_pos] = bracket_g[low_pos].clone();
                bracket_gtd[high_pos] = bracket_gtd[low_pos];
            }
            bracket[low_pos] = t;
            bracket_f[low_pos] = f_new;
            bracket_g[low_pos] = g_new.clone();
            bracket_gtd[low_pos] = gtd_new;
        }
    }
    let i = if bracket.len() == 1 { 0 } else { low_pos };
    (bracket_f[i], bracket_g[i].clone(), bracket[i], ls_evals)
}

/// Minimise obj in place on x; returns (final f, n_evals).
pub fn lbfgs<F: FnMut(&[f32]) -> (f32, Vec<f32>)>(
    mut obj: F,
    x: &mut [f32],
    opt: &LbfgsOptions,
) -> (f32, usize) {
    let (f0, mut g) = obj(x);
    let mut f = f0 as f64;
    let mut n_eval = 1usize;
    let mut n_iter = 0usize;
    let gmax = |g: &[f32]| g.iter().fold(0.0f64, |m, v| m.max(v.abs() as f64));
    if gmax(&g) <= opt.tolerance_grad as f64 {
        return (f as f32, n_eval);
    }
    let mut s_hist: VecDeque<Vec<f32>> = VecDeque::new();
    let mut y_hist: VecDeque<Vec<f32>> = VecDeque::new();
    let mut ro: VecDeque<f64> = VecDeque::new();
    let mut h_diag = 1.0f64;
    let mut d: Vec<f32> = g.iter().map(|v| -v).collect();
    let mut prev_g: Vec<f32> = g.clone();
    let mut t;
    let mut first = true;
    loop {
        n_iter += 1;
        if !first {
            let y: Vec<f32> = g.iter().zip(&prev_g).map(|(a, b)| a - b).collect();
            let s: Vec<f32> = d.iter().map(|v| v * 1.0).collect(); // d already scaled by t below
            let ys = dot(&y, &s);
            if ys > 1e-10 {
                if s_hist.len() == opt.history {
                    s_hist.pop_front();
                    y_hist.pop_front();
                    ro.pop_front();
                }
                h_diag = ys / dot(&y, &y);
                s_hist.push_back(s);
                y_hist.push_back(y);
                ro.push_back(1.0 / ys);
            }
            // two-loop recursion
            let m = s_hist.len();
            let mut q: Vec<f64> = g.iter().map(|v| -(*v as f64)).collect();
            let mut al = vec![0.0f64; m];
            for i in (0..m).rev() {
                al[i] = s_hist[i]
                    .iter()
                    .zip(&q)
                    .map(|(a, b)| *a as f64 * b)
                    .sum::<f64>()
                    * ro[i];
                for (qj, yj) in q.iter_mut().zip(&y_hist[i]) {
                    *qj -= al[i] * *yj as f64;
                }
            }
            let mut r: Vec<f64> = q.iter().map(|v| v * h_diag).collect();
            for i in 0..m {
                let be = y_hist[i]
                    .iter()
                    .zip(&r)
                    .map(|(a, b)| *a as f64 * b)
                    .sum::<f64>()
                    * ro[i];
                for (rj, sj) in r.iter_mut().zip(&s_hist[i]) {
                    *rj += (al[i] - be) * *sj as f64;
                }
            }
            d = r.iter().map(|v| *v as f32).collect();
        }
        prev_g = g.clone();
        let gtd = dot(&g, &d);
        if gtd > -opt.tolerance_change as f64 {
            break;
        }
        t = if first {
            (1.0f64).min(1.0 / g.iter().map(|v| v.abs() as f64).sum::<f64>()) * opt.lr as f64
        } else {
            opt.lr as f64
        };
        let (f_new, g_new, t_used, ls_evals) = strong_wolfe(
            &mut obj,
            x,
            t,
            &d,
            f,
            &g,
            gtd,
            1e-4,
            0.9,
            opt.tolerance_change as f64,
            25,
        );
        n_eval += ls_evals;
        // apply step and store scaled direction for the next curvature pair
        for (xi, di) in x.iter_mut().zip(&d) {
            *xi += (t_used as f32) * di;
        }
        for di in d.iter_mut() {
            *di *= t_used as f32;
        }
        let f_prev = f;
        f = f_new;
        g = g_new;
        first = false;
        if n_iter == opt.max_iter || n_eval >= opt.max_eval || gmax(&g) <= opt.tolerance_grad as f64
        {
            break;
        }
        if d.iter().fold(0.0f64, |m, v| m.max(v.abs() as f64)) <= opt.tolerance_change as f64 {
            break;
        }
        if (f - f_prev).abs() < opt.tolerance_change as f64 {
            break;
        }
    }
    (f as f32, n_eval)
}
