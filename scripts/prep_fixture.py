"""Derive unpile's inputs from a TRXScan fixture directory into <fixture>/unpile/.

Writes b0_<dir>.nii.gz (mean of the b<50 volumes), clean_b0.nii.gz and brain_mask.nii.gz (from the
simulator's clean b=0 truth), truth_field.nii.gz (Hz), T2w.nii.gz / T1w.nii.gz resampled onto the EPI
grid, gre_hz.nii.gz (the simulated phase difference unwrapped and converted to Hz on the EPI grid,
when the recipe has a GRE) and meta.json (readout time, polarity, PE axis, truth transform paths).
"""
import glob
import json
import os
import sys

import nibabel as nb
import numpy as np
from nilearn.image import resample_to_img
from scipy import ndimage as ndi


def main(root):
    out = os.path.join(root, 'unpile')
    os.makedirs(out, exist_ok=True)
    sub = os.path.basename(sorted(glob.glob(f'{root}/sub-*'))[0])
    deriv = f'{root}/derivatives/trxscan/{sub}'
    meta, ref = {}, None
    for f in sorted(glob.glob(f'{root}/{sub}/dwi/{sub}_dir-*_dwi.nii.gz')):
        d = f.split('_dir-')[1].split('_')[0]
        img = nb.load(f)
        bval = np.loadtxt(f[:-7] + '.bval')
        b0 = np.asarray(img.dataobj)[..., bval < 50].mean(-1).astype(np.float32)
        nb.Nifti1Image(b0, img.affine).to_filename(f'{out}/b0_{d}.nii.gz')
        side = json.load(open(f[:-7] + '.json'))
        ped = side['PhaseEncodingDirection']
        meta[d] = {'trt': side['TotalReadoutTime'], 'pe': ped, 'sign': -1.0 if ped.endswith('-') else 1.0, 'pe_axis': 'ijk'.index(ped[0]), 'voxel_mm': [float(z) for z in img.header.get_zooms()[:3]]}
        ref = ref or nb.Nifti1Image(b0, img.affine)
    c = nb.load(glob.glob(f'{deriv}/dwi/*desc-cleanb0*.nii.gz')[0])
    cd = np.asarray(c.dataobj, np.float32).squeeze()
    cd = cd[..., 0] if cd.ndim == 4 else cd
    nb.Nifti1Image(cd, c.affine).to_filename(f'{out}/clean_b0.nii.gz')
    m = ndi.binary_opening(ndi.binary_fill_holes(cd > 0.15 * np.percentile(cd[cd > 0], 99)), iterations=1)
    nb.Nifti1Image(m.astype(np.uint8), c.affine).to_filename(f'{out}/brain_mask.nii.gz')
    t = nb.load(glob.glob(f'{deriv}/dwi/*desc-fieldmap*.nii.gz')[0])
    nb.Nifti1Image(np.asarray(t.dataobj, np.float32).squeeze(), t.affine).to_filename(f'{out}/truth_field.nii.gz')
    for anat in ('T2w', 'T1w'):
        fs = glob.glob(f'{root}/{sub}/anat/{sub}_{anat}.nii.gz')
        if fs:
            resample_to_img(nb.load(fs[0]), ref, interpolation='continuous').to_filename(f'{out}/{anat}.nii.gz')
    meta['truth_xfm'] = {}
    for k in ('T2w', 'T1w', 'fmap'):
        fs = glob.glob(f'{deriv}/*/{sub}_*from-{k}_to-dwi*desc-truth_xfm.txt')
        meta['truth_xfm'][k] = os.path.relpath(fs[0], out) if fs else None
    pd = glob.glob(f'{root}/{sub}/fmap/*_phasediff.nii.gz')
    if pd:
        from skimage.restoration import unwrap_phase
        img = nb.load(pd[0])
        x = np.asarray(img.dataobj, np.float64)
        side = json.load(open(pd[0][:-7] + '.json'))
        if x.max() - x.min() > 2 * np.pi + 0.1:  # stored as scaled integers, Siemens style
            x = (x - x.min()) / (x.max() - x.min()) * 2 * np.pi - np.pi
        mag = sorted(glob.glob(f'{root}/{sub}/fmap/*_magnitude1.nii.gz'))
        md = np.asarray(nb.load(mag[0]).dataobj, np.float32)
        mk = md > 0.15 * np.percentile(md, 99)
        u = np.asarray(unwrap_phase(np.ma.array(x, mask=~mk)))
        u -= 2 * np.pi * np.round(np.median(u[mk]) / (2 * np.pi))  # the unwrapper's arbitrary 2*pi*k
        dte = abs(side['EchoTime2'] - side['EchoTime1'])
        hz = (u / (2 * np.pi * dte)).astype(np.float32)
        hz[~mk] = 0
        resample_to_img(nb.Nifti1Image(hz, img.affine), ref, interpolation='continuous').to_filename(f'{out}/gre_hz.nii.gz')
        meta['gre'] = {'delta_te_s': dte, 'phasediff': os.path.relpath(pd[0], out)}
    json.dump(meta, open(f'{out}/meta.json', 'w'), indent=1)
    print(f'{out}: ' + ', '.join(sorted(os.listdir(out))))


if __name__ == '__main__':
    main(sys.argv[1])
