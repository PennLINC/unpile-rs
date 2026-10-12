//! Phase-encode-last layout: the estimator works on arrays whose LAST axis is the PE axis.
use ndarray::{Array3, Axis};

pub fn to_pe_last(a: &Array3<f32>, pe_axis: usize) -> Array3<f32> {
    let mut v = a.view();
    // move pe_axis to the end, keeping the other two in order
    let order: Vec<usize> = (0..3)
        .filter(|&x| x != pe_axis)
        .chain(std::iter::once(pe_axis))
        .collect();
    v = v.permuted_axes([order[0], order[1], order[2]]);
    v.as_standard_layout().to_owned()
}

pub fn from_pe_last(a: &Array3<f32>, pe_axis: usize) -> Array3<f32> {
    let order: Vec<usize> = (0..3)
        .filter(|&x| x != pe_axis)
        .chain(std::iter::once(pe_axis))
        .collect();
    let mut inv = [0usize; 3];
    for (new_pos, &old) in order.iter().enumerate() {
        inv[old] = new_pos;
    }
    a.view().permuted_axes(inv).as_standard_layout().to_owned()
}

pub fn spacing_pe_last(vox: [f32; 3], pe_axis: usize) -> [f32; 3] {
    let others: Vec<f32> = (0..3).filter(|&x| x != pe_axis).map(|x| vox[x]).collect();
    [others[0], others[1], vox[pe_axis]]
}

pub fn mean_in(a: &Array3<f32>, mask: &Array3<bool>) -> f32 {
    let mut s = 0.0f64;
    let mut n = 0usize;
    for (v, &m) in a.iter().zip(mask.iter()) {
        if m {
            s += *v as f64;
            n += 1;
        }
    }
    (s / n.max(1) as f64) as f32
}

pub fn median_in(a: &Array3<f32>, mask: &Array3<bool>) -> f32 {
    let mut v: Vec<f32> = a
        .iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(x, _)| *x)
        .collect();
    if v.is_empty() {
        return 1.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Binary dilation by `iters` steps of 6-connectivity (matches scipy's default structuring element).
pub fn dilate(mask: &Array3<bool>, iters: usize) -> Array3<bool> {
    let mut cur = mask.clone();
    let (n0, n1, n2) = cur.dim();
    for _ in 0..iters {
        let prev = cur.clone();
        for i in 0..n0 {
            for j in 0..n1 {
                for k in 0..n2 {
                    if prev[[i, j, k]] {
                        continue;
                    }
                    let hit = (i > 0 && prev[[i - 1, j, k]])
                        || (i + 1 < n0 && prev[[i + 1, j, k]])
                        || (j > 0 && prev[[i, j - 1, k]])
                        || (j + 1 < n1 && prev[[i, j + 1, k]])
                        || (k > 0 && prev[[i, j, k - 1]])
                        || (k + 1 < n2 && prev[[i, j, k + 1]]);
                    if hit {
                        cur[[i, j, k]] = true;
                    }
                }
            }
        }
    }
    cur
}

pub fn erode(mask: &Array3<bool>, iters: usize) -> Array3<bool> {
    let inv = mask.map(|&m| !m);
    dilate(&inv, iters).map(|&m| !m)
}

pub fn _unused(_: Axis) {}
