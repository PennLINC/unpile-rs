# How the fit works, on one line

unpile estimates the field by predicting the acquired EPI from an undistorted image and changing
the field until the prediction matches. This page follows that fit on a single line of voxels along
the phase-encoding axis, from a zero field to the answer. Every figure was made with the short
PyTorch versions of unpile's operators in `docs/tools/line_model.py`. Before drawing, the figure
script checks those against the PyTorch prototype's stored outputs, the same files `unpile check`
holds the Rust code to. On every line of that volume they agree to 4e-6.

The data are the simulated test set in `testdata/rpe`: a 3 mm b=0 with phase encoding along the
anterior-posterior axis, a total readout time of 55.7 ms, the clean undistorted b=0 it was made
from, and the true field. The line runs from back to front through the middle of the brain, at the
level of the frontal hot spot where the field reaches 102 Hz.

## The line

```{figure} ../_static/line_setup.png
:alt: The undistorted and acquired b=0 with the line marked, the signal along the line in both, and the true displacement along it.

Left: the undistorted reference and the acquired b=0, with the line marked. Right, top: the signal
along the line (grey: inside the brain). Right, bottom: the true displacement, with the compressed
(red) and stretched (blue) stretches shaded.
```

During the readout, signal at a point where the field is $f$ Hz lands $f \times$ readout time
voxels away from where it belongs. Here 1 Hz is 0.056 voxel, or 0.17 mm. If every voxel on the line
moved by the same amount the image would just shift. The trouble comes from the change in
displacement along the line:

- From voxel 40 to 55 the displacement falls from +1 to −17 mm, so each voxel moves further
  backwards than the one behind it. Neighbouring voxels are squeezed together: around voxel 45,
  three voxels' worth of signal lands in about one. That is the bright peak in the acquired b=0,
  2.4 times the reference's brightest voxel nearby. This is **pile-up**.
- From voxel 68 to 76 the displacement climbs back to zero, so voxels are pulled apart and their
  signal is spread thin. This is **stretching**; in the extreme it becomes dropout.

The total signal on the line does not change. It is only moved around.

## Step 1: predict the acquired line

The forward model writes that last sentence as code. Picture each voxel of the undistorted
reference as a cell one voxel wide whose **area** is its signal. Move every cell edge by the
displacement at that edge, keep each cell's area, and then add up how much area lands inside each
voxel:

```{figure} ../_static/line_cells.png
:alt: Coloured cells for each reference voxel, the same cells after their edges moved, and the stacked contributions each output voxel collects.

The reference's voxels as cells (top); the same cells with their edges moved by the true
displacement, each keeping its area, so squeezed cells grow taller (middle); and what each voxel of
the prediction collects, stacked by the cell it came from (bottom). The black dots are the acquired
b=0. Voxel 44, the top of the peak, collects most of reference voxels 46 to 49.
```

That is the whole forward model: a pushforward that conserves signal, which is why unpile calls
it a *splat*. In the code a cell is the interval between its two moved edges, and the share of
cell $i$ that lands in voxel $j$ is one entry of a matrix:

```{literalinclude} ../tools/line_model.py
:pyobject: moved_cells
```

```{literalinclude} ../tools/line_model.py
:pyobject: splat_weights
```

One detail makes the fit possible. With sharp voxel boundaries, a cell narrower than a voxel adds
the same amount wherever it sits inside that voxel. Nudging it then changes nothing, so the fit has
no gradient to follow. unpile blurs each voxel boundary over half a voxel (the `ramp` function, a
tent by default), so the prediction always changes smoothly as an edge moves.

## Why predict forward instead of undoing the distortion

The alternative is to go the other way: resample the acquired image at the displaced positions to
undo the distortion, and compare the result with the reference. A general-purpose registration
does that. TOPUP's model does it too, and also scales the intensities by the Jacobian (the local
stretch, $1 + d'$).

```{figure} ../_static/line_push_pull.png
:alt: At the true field, the reference pushed forward against the acquired line, and the acquired line pulled back against the reference.

Both directions at the true field, on the stretch of the line around the pile-up.
```

Both are exact in a noiseless continuous world. On a voxel grid they are not. In the pile-up, one
acquired voxel holds what came from three reference voxels. Pulling it back can only spread that
one value over the three, so the detail between them is lost. Pushing forward sums the detailed
reference and loses nothing the comparison needs. In the 16 pile-up voxels of this line, at the
true field:

| comparison | rms difference |
|---|---|
| reference pushed forward vs the acquired b=0 | 0.12 |
| acquired b=0 pulled back × Jacobian vs the reference | 0.18 |
| acquired b=0 resampled only vs the reference | 1.55 |

The last row is the important one for registration-based correction. Plain resampling cannot make
a voxel dimmer. A registration that only resamples can explain a pile-up peak only by bending the
field out of shape. In stretched voxels the advantage reverses slightly (0.08 for the push against
0.06 for the pull), so the push model's gain is where the signal piles up, which is where
fieldmap-less methods have the most trouble.

## Step 2: score a guess

For a candidate field, unpile predicts the acquired image and measures how far off it is. With a
same-contrast reference (the clean b=0 here, or a SynB0 image):

$$
\text{loss} = \underbrace{\frac{1}{|M|}\sum_{y \in M} \big(g\,\text{pred}(y) - \text{obs}(y)\big)^2}_{\text{data term}}
\; + \; \lambda \, \underbrace{\overline{\left(\partial^2 f\right)^2}}_{\text{bending energy}}
\; + \; \underbrace{\overline{f^2}\big|_{\text{outside the head}}}_{\text{background}}
$$

- $M$ is the brain mask dilated by four voxels.
- $g$ is one intensity gain, solved in closed form at every evaluation, because the reference and
  the EPI rarely have the same scale.
- The bending energy, the mean squared second derivative of the field, keeps the field smooth.
  $\lambda$ is `--lam` (0.1 by default) or `--lam-mm`.
- The last term pulls the field to zero more than three voxels outside the head, where there is no
  signal to pin it.

With a T2w reference, the data term is instead one minus the local normalized cross-correlation in
5-voxel windows (`--metric lncc`), which compares shapes rather than intensities.

## Step 3: which way to move

At the start the field is zero, so the prediction is just the reference. PyTorch differentiates
the loss through the splat and gives the direction in which to change the field at every voxel:

```{figure} ../_static/line_gradient.png
:alt: The prediction with no field against the acquired line, and the downhill direction for the field, raw and through the coarse control grid, next to the true displacement.

Top: with no field, the prediction (dashed) misses the pile-up peak and has its edges in the wrong
places. Bottom: the downhill direction for the field at each voxel (orange); the same step after
it passes through the 24 mm control grid (blue); and the true displacement for comparison (grey).
All three are scaled to the same height.
```

The raw gradient is local. It is large only where an edge of the prediction sits in the wrong
place relative to the acquisition, around voxels 42–48 and 63–69, and it says almost nothing in between.
That alone would be a poor guide. unpile does not fit the field voxel by voxel, though. The field
is a cubic B-spline whose control points start 24 mm (8 voxels) apart, and each control point
gathers the local pushes from the 32 voxels its spline covers. Through the coarse grid, the very first step already has roughly the
shape of the true displacement.

## Step 4: iterate

unpile minimises the loss over the control-point values with L-BFGS, a quasi-Newton method that
builds up a picture of the loss's curvature from the gradients it has seen. `src/lbfgs.rs` follows
`torch.optim.LBFGS` with a strong-Wolfe line search, which is what the walkthrough uses.

```{figure} ../_static/line_fit.gif
:alt: Animation: the prediction converging on the acquired line while the fitted displacement converges on the true one.

The whole fit, one frame per iteration (every sixth late in each level).
```

```{figure} ../_static/line_fit_steps.png
:alt: Snapshots of prediction against acquisition and fitted against true displacement at iterations 0, 1, 3, 10 and 60 of the coarse grid and at the end.

Snapshots. Left: the prediction (dashed, × gain) against the acquired b=0. Right: the fitted
displacement against the truth, with the rms error inside the brain.
```

The first iteration only moves cautiously in the right direction (9.2 → 7.8 mm rms error). By the
third the pile-up peak has started to form in the prediction (2.6 mm). By the tenth the peak is
in place (1.3 mm). The coarse grid then settles at 1.1 mm: 24 mm control points cannot follow the
field's sharper turns.

## Step 5: coarse to fine

The default schedule fits on control grids of 24, 12, 6 and 3 mm (`--levels`), each starting from
the previous one's field. On all but the last, the prediction and the acquisition are both blurred
by a Gaussian of half the grid spacing (up to 3 voxels). The fit therefore first matches the broad
shape of the signal, where a large error in the field still shows up as a smooth error in the
images, and only then matches the fine detail.

```{figure} ../_static/line_levels.png
:alt: The data term and bending energy over all iterations of all levels, and the fitted displacement after each level against the truth.

Left: the data term and the weighted bending energy over the whole schedule. The data term jumps
at the start of each level because the blur gets smaller, so the comparison gets stricter. Right:
the displacement after each level against the truth.
```

On this line the error goes 1.12, 0.53, 0.36, 0.44 mm. The 3 mm grid fits a little of the noise,
because on a single line nothing ties the field to its neighbours. In the volume the bending energy
couples every line to the ones beside and above it.

## Step 6: the corrected line

Once the field is known, the corrected image is the one that, pushed forward, best explains the
acquisition. unpile solves for it by least squares with a small smoothness penalty (conjugate
gradients in `unwarp`) and writes it as `corrected.nii.gz`.

```{figure} ../_static/line_unwarp.png
:alt: Corrected lines from the fitted field: resampled only, resampled with the Jacobian, and the inverse of the push, against the reference.

The corrected line around the pile-up, divided by the fitted gain to put it on the reference's
scale.
```

The ITK warp, `sdc_warp.nii.gz`, is what qsiprep and `antsApplyTransforms` use. It moves signal
back to where it belongs, but resampling cannot undo a change in brightness: the pile-up comes back
as a broad bright band (dotted). Scaling by the Jacobian fixes the brightness (dashed). The inverse
of the push also recovers the detail inside the pile-up. Against the reference, in the pile-up
voxels, the rms differences are 1.52 resampled only, 0.16 with the Jacobian and 0.09 for the
inverse of the push.

## From one line to the volume

The real fit is the same objective in three dimensions:

- The field is one 3-D cubic B-spline. Every line along the phase-encoding axis is pushed forward
  independently, but the lines share control points and the bending energy includes every axis and
  the cross terms, so neighbouring lines constrain each other.
- With a T2w reference (`--metric lncc --pose`) a rigid position of the T2w is fitted together with
  the field. A uniform shift along the phase-encoding axis could be either a field offset or a
  translation, so the field's mean inside the brain is held near zero.
- With a gradient-echo fieldmap (`--fieldmap`) the field is the fieldmap under its own rigid
  position, plus a second-order shim polynomial, plus a B-spline residual kept small by a ridge
  penalty. The same splat and data term fit all three. See {doc}`gre-refinement`.
- With an opposite-polarity pair (`--reverse-epi`) there is no reference. The undistorted image is
  re-solved between rounds as the one that, pushed forward with $+d$ and $-d$, best explains both
  acquisitions. See {doc}`opposite-polarity`.

## Run it yourself

```bash
pip install torch nibabel matplotlib
python docs/tools/make_line_walkthrough.py --testdata testdata/rpe --out docs/_static
```

The script prints the operator check, then the fit level by level. It differs from `unpile fit` in
small ways that do not change the story: the bending energy and the blur act along the line only,
the corrected line comes from a direct solve rather than 25 conjugate-gradient iterations, and
$\lambda$ is the CLI default, not tuned for one dimension. The walkthrough uses the tent kernel that
`unpile fit` uses. The operator check uses the prototype's Gaussian kernel, because the stored
outputs were made with it.
