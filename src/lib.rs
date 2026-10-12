#![allow(
    clippy::single_range_in_vec_init,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::identity_op,
    clippy::erasing_op
)]
pub mod field;
pub mod io;
pub mod latent;
pub mod layout;
pub mod lbfgs;
pub mod ops;
pub mod optimize;
pub mod pose;
pub mod problem;
pub mod profile;
pub mod score;
pub mod xfm;
