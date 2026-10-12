# Opposite-polarity pairs

Two images with reversed phase-encoding polarity contain the field between them: what is
compressed in one is stretched in the other ([see it in the book](https://pennlinc.github.io/diffusion-book/book/preprocessing/susceptibility-distortion#see-it-the-blip-up-blip-down-pair)).
FSL topup and TORTOISE DRBUDDI are the established tools for this case and qsiprep uses them.
unpile supports it so that one binary covers every input a dataset may provide:

```bash
unpile fit --obs b0_AP.nii.gz --pa b0_PA.nii.gz --mask brain.nii.gz \
    --trt 0.0557 --pe-axis 1 --sign=-1 --lam-mm 5 --out out
```

The field is estimated through a latent undistorted image that both acquisitions are predicted
from; `--blip-pose` and `--blip-gain` add a rigid pose and an intensity gain for the second image
when the subject moved or the scaling differs between the two. The latent image is written as
`corrected.nii.gz`, with `corrected_pa.nii.gz` for the second polarity.

```{figure} ../_static/two_blip.png
:alt: The two acquisitions, the corrected image and the simulated truth.

The simulated `rpe` fixture: 0.987 field correlation with the truth at the test schedule.
```
