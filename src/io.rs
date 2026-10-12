//! NIfTI I/O: volumes as ndarray Array3<f32> in the file's own axis order (i, j, k).
use anyhow::{Context, Result};
use ndarray::{Array3, ArrayD, Axis, Ix3, Ix4};
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions};
use std::path::Path;

pub struct Volume {
    pub data: Array3<f32>,
    pub header: NiftiHeader,
}

/// Read a 3D volume; a 4D file is averaged over its last axis unless `vols` selects indices.
pub fn read3(path: impl AsRef<Path>, vols: Option<&[usize]>) -> Result<Volume> {
    let obj = ReaderOptions::new()
        .read_file(path.as_ref())
        .with_context(|| format!("reading {:?}", path.as_ref()))?;
    let header = obj.header().clone();
    let arr: ArrayD<f32> = obj.into_volume().into_ndarray::<f32>()?;
    let data = match arr.ndim() {
        3 => arr.into_dimensionality::<Ix3>()?,
        4 => {
            let a4 = arr.into_dimensionality::<Ix4>()?;
            let n = a4.len_of(Axis(3));
            let sel: Vec<usize> = vols.map(|v| v.to_vec()).unwrap_or_else(|| (0..n).collect());
            let mut acc =
                Array3::<f32>::zeros((a4.len_of(Axis(0)), a4.len_of(Axis(1)), a4.len_of(Axis(2))));
            for &v in &sel {
                acc += &a4.index_axis(Axis(3), v);
            }
            acc / sel.len() as f32
        }
        d => anyhow::bail!("expected a 3D or 4D image, got {d}D"),
    };
    Ok(Volume {
        data: data.as_standard_layout().to_owned(),
        header,
    })
}

pub fn write3(path: impl AsRef<Path>, data: &Array3<f32>, reference: &NiftiHeader) -> Result<()> {
    nifti::writer::WriterOptions::new(path.as_ref())
        .reference_header(reference)
        .write_nifti(data)?;
    Ok(())
}

/// Fail unless `other` is on the grid of `reference` (same shape and voxel-to-world affine).
/// Inputs are taken as voxel arrays, so a different orientation or offset would be silently wrong.
pub fn check_same_grid(reference: &Volume, other: &Volume, name: &str) -> Result<()> {
    if reference.data.dim() != other.data.dim() {
        anyhow::bail!(
            "{name} has shape {:?}, the observed image {:?}; resample it onto the EPI grid first",
            other.data.dim(),
            reference.data.dim()
        );
    }
    let (a, b) = (
        reference.header.affine::<f64>(),
        other.header.affine::<f64>(),
    );
    let maxdiff = (0..4)
        .flat_map(|i| (0..4).map(move |j| (i, j)))
        .map(|(i, j)| (a[(i, j)] - b[(i, j)]).abs())
        .fold(0.0, f64::max);
    if maxdiff > 1e-2 {
        anyhow::bail!("{name} is not on the grid of the observed image (affines differ by up to {maxdiff:.3} mm); resample it onto the EPI grid first");
    }
    Ok(())
}

/// Voxel sizes (mm) from the header.
pub fn zooms(h: &NiftiHeader) -> [f32; 3] {
    [h.pixdim[1].abs(), h.pixdim[2].abs(), h.pixdim[3].abs()]
}
