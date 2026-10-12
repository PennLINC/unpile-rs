//! Wall-clock per-operation accounting for the eager CPU backend (`UNPILE_PROFILE=1`).
//! Scopes are inclusive (a `pullback` scope contains its `interp_last`); the wgpu backend
//! executes lazily, so its numbers only show where the host waited.
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static ACC: OnceLock<Mutex<HashMap<&'static str, (u64, Duration)>>> = OnceLock::new();

pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var_os("UNPILE_PROFILE").is_some())
}

pub fn scope<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let r = f();
    let d = t.elapsed();
    let mut m = ACC.get_or_init(Default::default).lock().unwrap();
    let e = m.entry(name).or_insert((0, Duration::ZERO));
    e.0 += 1;
    e.1 += d;
    r
}

pub fn report() {
    if !enabled() {
        return;
    }
    let m = ACC.get_or_init(Default::default).lock().unwrap();
    let mut rows: Vec<_> = m.iter().collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1 .1));
    println!("{:<22} {:>8} {:>10} {:>9}", "scope (inclusive)", "calls", "total s", "mean ms");
    for (name, (n, d)) in rows {
        println!("{:<22} {:>8} {:>10.2} {:>9.2}", name, n, d.as_secs_f64(), d.as_secs_f64() * 1000.0 / *n as f64);
    }
}
