# Opposite-polarity pairs

Two images with reversed phase-encoding direction contain the field between them: what is
compressed in one is stretched in the other ([see it in the book](https://pennlinc.github.io/diffusion-book/book/preprocessing/susceptibility-distortion#see-it-the-blip-up-blip-down-pair)).
FSL topup and TORTOISE DRBUDDI are the established tools for this, and qsiprep uses them. unpile
supports it too, so one binary covers every input a dataset may have:

```bash
unpile fit --epi b0_AP.nii.gz --reverse-epi b0_PA.nii.gz --mask brain.nii.gz \
    --pe-dir j- --total-readout-time 0.0557 --lam-mm 5 --out out
```

`--pe-dir` describes the first image; the second has the opposite direction. The field is estimated
through an undistorted image that both acquisitions are predicted from. `--blip-pose` and
`--blip-gain` add a rigid position and an intensity gain for the second image. The undistorted
image is written as `corrected.nii.gz`, with `corrected_pa.nii.gz` for the second direction.

```{figure} ../_static/two_blip.png
:alt: The two acquisitions, the corrected image and the simulated truth.

The simulated `rpe` dataset: 0.987 field correlation with the truth at the test schedule.
```
