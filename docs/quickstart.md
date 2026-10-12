# Quickstart

You need an EPI image, its BIDS sidecar, a brain mask and a reference image. The examples use the
test data bundle, which has all of these:

```bash
scripts/fetch_testdata.sh
ls testdata/real/v1/ds002737
```

This is a 2.4 mm single-band reference image from OpenNeuro ds002737 with the subject's T2w and a
brain mask on the same grid. The sidecar values are in `meta.json`:

```json
"PhaseEncodingDirection": "j-",
"TotalReadoutTime": 0.0525111
```

## Fieldmap-less correction with a T2w

```bash
D=testdata/real/v1/ds002737
unpile fit --epi $D/sbref_ap.nii.gz --reference $D/t2w.nii.gz --mask $D/brain_mask.nii.gz \
    --pe-dir j- --total-readout-time 0.0525111 \
    --metric lncc --pose --lam-mm 60 --out out/sbref
```

`--metric lncc` compares the two contrasts with local correlation. `--pose` fits a rigid position of
the T2w at the same time as the field. The run takes about 40 s on a GPU and 6 minutes on four CPU
cores. `out/sbref/` contains the field in Hz, the displacement in mm, the corrected image, the ITK
warp and the T2w pose as an ITK affine.

```{figure} _static/fieldmapless_sbref_before_after.svg
:alt: The single-band reference before and after correction, alternating, with contours of the T2w's bright tissue.

The image before and after correction, alternating. The red contours are the bright tissue of the T2w.
This run was fitted on a 1.8 mm grid; see {doc}`user/fieldmapless` for why.
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
