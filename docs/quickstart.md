# Quickstart

You need an EPI image, its BIDS sidecar, a brain mask and a reference image on the same grid. The
examples use the test data bundle, which has all of these:

```bash
scripts/fetch_testdata.sh
ls testdata/real/v1/bold5000/ses-01
```

This is a 2 mm spin-echo EPI from OpenNeuro BOLD5000 with the subject's T2w and a brain mask on its
grid. The sidecar values are in `meta.json`:

```json
"PhaseEncodingDirection": "j-",
"TotalReadoutTime": 0.0755984
```

## Fieldmap-less correction with a T2w

```bash
D=testdata/real/v1/bold5000/ses-01
unpile fit --epi $D/se_ap.nii.gz --reference $D/t2w.nii.gz --mask $D/brain_mask.nii.gz \
    --pe-dir j- --total-readout-time 0.0755984 \
    --metric lncc --pose --lam-mm 60 --out out/se
```

`--metric lncc` compares the two contrasts with local correlation. `--pose` fits a rigid position of
the T2w at the same time as the field. The run takes about 30 s on a GPU and a few minutes on four
CPU cores. `out/se/` contains the field in Hz, the displacement in mm, the corrected image, the ITK
warp and the T2w pose as an ITK affine.

```{figure} _static/fieldmapless_before_after.svg
:alt: The spin-echo EPI before and after correction, alternating, with the white-matter boundary from the T2w.

The image before and after correction, alternating every two seconds, through the slice where the
displacement is largest. The red line is the white-matter boundary from a SynthSeg segmentation of
the T2w. The field agrees with FSL topup's from the session's spin-echo pair at 0.85 correlation.
When the T2w has smaller voxels than the EPI, fitting on a finer grid helps; see {doc}`user/fieldmapless`.
```

## Refining a fieldmap

With a fieldmap from the same session, pass it in Hz on the EPI grid and give a reference as well.
The example uses a simulated dataset from TRXScan, which has a b=0, a T2w and a GRE fieldmap:

```bash
pip install "trxscan[trx]==0.2.1" nibabel nilearn scikit-image scipy
python scripts/make_fixtures.py phasediff
D=testdata/fixtures/phasediff/unpile
unpile fit --epi $D/b0_AP.nii.gz --reference $D/T2w.nii.gz --mask $D/brain_mask.nii.gz \
    --pe-dir j- --total-readout-time 0.055675 \
    --fieldmap $D/gre_hz.nii.gz --fieldmap-refine pose,shim,delta \
    --metric lncc --pose --lam-mm 60 --out out/fieldmap
```

The log reports the position unpile found for the fieldmap. On this dataset the fieldmap was
acquired after a 5.4° / 4 mm movement, and the fit recovers it to 0.8° / 0.6 mm.

```{figure} _static/gre_refinement_before_after.svg
:alt: The simulated b=0 before and after correction with the refined fieldmap, alternating, with the white-matter boundary.

The simulated b=0 before and after correction with the refined fieldmap, alternating every two
seconds. The red line is the white-matter boundary of the simulated subject. The displacement is
2.7 mm at the median voxel and up to 17 mm at the front of the brain.
```

## Applying the correction

The warp is a standard ITK displacement field:

```bash
antsApplyTransforms -d 3 -i epi.nii.gz -r epi.nii.gz -t out/sbref/sdc_warp.nii.gz \
    -o epi_unwarped.nii.gz -n BSpline
```

The warp moves the signal back. The intensity change from pile-up and stretching is a separate
multiplication by the Jacobian of the displacement, which qsiprep applies with its own module.
{doc}`user/inputs-outputs` has the conventions.

## Preparing your own data

unpile does not make the brain mask, resample the anatomical image, or unwrap a fieldmap; qsiprep
does these before calling it. For a stand-alone run:

- Brain mask: SynthStrip or `niimath -skullstrip` on the EPI.
- T2w on the EPI grid: a rigid registration with `antsRegistration`, then `antsApplyTransforms -r epi.nii.gz`.
- Fieldmap in Hz on the EPI grid: unwrap the phase difference, divide by 2π ΔTE, register the
  magnitude image to the EPI and resample.

Every image must share the EPI's grid; unpile checks and refuses otherwise.
