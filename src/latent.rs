//! Least-squares restoration for the two-blip mode: the latent (undistorted) image that, pushed
//! forward with +-d, best explains both blips; CG on the normal equations with the adjoint of the
//! (linear) forward operator obtained by autodiff.
use burn::tensor::Tensor;

/// -Laplacian (unit spacing) with replicate ends: the operator of ||grad x||^2.
pub fn laplacian_neg(x: Tensor<3>) -> Tensor<3> {
    let dims = x.dims();
    let mut out: Option<Tensor<3>> = None;
    for ax in 0..3 {
        let n = dims[ax];
        let mut r_first = [0..dims[0], 0..dims[1], 0..dims[2]];
        r_first[ax] = 0..1;
        let mut r_last = [0..dims[0], 0..dims[1], 0..dims[2]];
        r_last[ax] = n - 1..n;
        let mut r_lo = [0..dims[0], 0..dims[1], 0..dims[2]];
        r_lo[ax] = 1..n;
        let mut r_hi = [0..dims[0], 0..dims[1], 0..dims[2]];
        r_hi[ax] = 0..n - 1;
        // xp[i] = x[i+1] (last replicated); xm[i] = x[i-1] (first replicated)
        let xp = Tensor::cat(vec![x.clone().slice(r_lo), x.clone().slice(r_last)], ax);
        let xm = Tensor::cat(vec![x.clone().slice(r_first), x.clone().slice(r_hi)], ax);
        let term = x.clone() * 2.0 - xp - xm;
        out = Some(match out {
            Some(o) => o + term,
            None => term,
        });
    }
    out.unwrap()
}

/// Adjoint of a linear operator by autodiff: op^T r = d/dx <op(x), r>.
pub fn adjoint<F: Fn(Tensor<3>) -> Tensor<3>>(op: F, like: &Tensor<3>, r: Tensor<3>) -> Tensor<3> {
    let x = Tensor::<3>::zeros(like.dims(), &like.device()).require_grad();
    let s = (op(x.clone()) * r).sum();
    let grads = s.backward();
    x.grad(&grads).expect("adjoint gradient").detach()
}

/// Minimise sum_k ||A_k x - y_k||^2 + ridge ||x||^2 + grad_reg ||grad x||^2 by CG.
pub fn cg_solve<A: Fn(Tensor<3>, usize) -> Tensor<3>, At: Fn(Tensor<3>, usize) -> Tensor<3>>(
    apply_a: A,
    apply_at: At,
    rhs: &[Tensor<3>],
    x0: Tensor<3>,
    n_iter: usize,
    ridge: f32,
    grad_reg: f32,
) -> Tensor<3> {
    let normal = |x: Tensor<3>| {
        let mut out = x.clone() * ridge;
        for k in 0..rhs.len() {
            out = out + apply_at(apply_a(x.clone(), k), k);
        }
        if grad_reg > 0.0 {
            out = out + laplacian_neg(x) * grad_reg;
        }
        out
    };
    let mut b: Option<Tensor<3>> = None;
    for (k, y) in rhs.iter().enumerate() {
        let t = apply_at(y.clone(), k);
        b = Some(match b {
            Some(acc) => acc + t,
            None => t,
        });
    }
    let b = b.unwrap();
    let mut x = x0;
    let mut r = b - normal(x.clone());
    let mut p = r.clone();
    let mut rs = (r.clone() * r.clone()).sum().into_scalar::<f32>();
    for _ in 0..n_iter {
        let ap = normal(p.clone());
        let pap = (p.clone() * ap.clone())
            .sum()
            .into_scalar::<f32>()
            .max(1e-20);
        let alpha = rs / pap;
        x = x + p.clone() * alpha;
        r = r - ap * alpha;
        let rs_new = (r.clone() * r.clone()).sum().into_scalar::<f32>();
        if rs_new.sqrt() < 1e-8 {
            break;
        }
        p = r.clone() + p * (rs_new / rs);
        rs = rs_new;
    }
    x
}
