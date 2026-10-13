# Estimation on a single line

unpile estimates the susceptibility field by forward modelling. Given an undistorted image and a
candidate field, it predicts the distorted EPI, compares the prediction with the acquired image,
and adjusts the field until the two agree. This page follows that procedure on one line of voxels
along the phase-encoding axis, from a zero field to the final estimate.

In unpile the computation runs on the whole volume at once: every line along the phase-encoding
axis is predicted with the same three-dimensional field, and a single objective summed over the
brain is minimised. Restricting it to one line here keeps every quantity small enough to plot. How
the line-wise computation extends to the volume is described at the end of the page.

The figures were produced with PyTorch implementations of unpile's one-dimensional operators
(`docs/tools/line_model.py`). These are checked against the outputs of the original PyTorch
prototype stored in `testdata/rpe`, which are also the reference for the Rust implementation
(`unpile check`). Over all lines of that volume the largest absolute difference is 4 × 10⁻⁶ of
the median brain signal.

## Data and conventions

The simulated data set in `testdata/rpe` contains a b=0 image with 3 mm isotropic voxels and
anterior–posterior phase encoding (total readout time 55.7 ms), the undistorted image from which it
was simulated, and the true field. The line runs from posterior to anterior through the midline of
an inferior slice that crosses the frontal susceptibility artefact. Along this line the field
reaches 102 Hz.

Intensities are scaled so that the median signal inside the brain is 1, and intensity errors are
reported as percentages of that median. Positions are voxel indices along the line, increasing
from posterior to anterior. Positive displacements point anteriorly.

## The distortion along the line

```{figure} ../_static/line_setup.svg
:alt: The undistorted and acquired b=0 with the line marked, the signal along the line in both, and the true displacement along it.

(A, B) The undistorted reference and the acquired b=0 in the slice containing the line (vertical
line). (C) Signal along the line; the grey band marks the brain. (D) True displacement along the
line. Shaded intervals mark where the displacement compresses the line (red) or stretches it
(blue) by more than 15%.
```

An off-resonance of $f$ Hz displaces signal along the phase-encoding axis by $f\,T_\text{ro}$
voxels, where $T_\text{ro}$ is the total readout time. In this acquisition 1 Hz corresponds to
0.056 voxels (0.17 mm), and the 102 Hz peak to a posterior displacement of 17 mm.

A uniform displacement would translate the image without changing its intensities. Intensities
change where the displacement varies along the line. A useful picture is a strip of elastic with
ink painted along it. Where the strip is compressed the ink is concentrated and appears darker;
where it is stretched the ink is spread out and appears paler. The total amount of ink does not
change. On this line:

- Between voxels 40 and 55 the displacement changes from 1 mm anterior to 17 mm posterior. Each
  voxel is displaced further posteriorly than its posterior neighbour, so neighbouring voxels
  converge. Near voxel 45 the line is compressed to 34% of its original length. The acquired
  signal there peaks at 5.2 times the median brain signal, compared with 2.2 for the brightest
  nearby voxel of the reference. This is *pile-up*.
- Between voxels 68 and 76 the displacement returns from 12 mm posterior to zero. Neighbouring
  voxels diverge and their signal is spread over a longer interval. This is *stretching*; in its
  extreme form it produces signal dropout.

## Forward model

unpile's forward model implements the elastic-strip picture. Each voxel of the reference is treated
as a cell of unit width whose area equals its signal. Both edges of each cell are displaced by the
displacement at that edge, taken as the mean of the displacements of the two voxels it separates,
so adjacent cells remain contiguous. Each cell keeps its area: a cell compressed to one third of
its width becomes three times as tall. The predicted value of an output voxel is the total area of
all cells that falls within it.

```{figure} ../_static/line_cells.svg
:alt: Coloured cells for each reference voxel, the same cells after their edges are displaced, and the contributions collected by each output voxel.

Voxels 34–61 of the line at the true displacement. (A) The reference, one cell per voxel. (B) The
same cells after their edges are displaced; each keeps its area, so compressed cells become taller.
(C) The prediction: the area collected by each output voxel, coloured by the cell it came from.
Black dots: the acquired b=0, divided by the fitted intensity gain (see Objective). The acquired
peak at voxel 44 receives most of its signal from reference voxels 46 to 49.
```

Because the operation redistributes area rather than interpolating values, it conserves the total
signal on the line. unpile refers to it as a *splat*, or mass-conserving pushforward. In matrix
form, the share of cell $i$ that falls in output voxel $j$ is the entry $W_{ij}$, and the prediction
is $\text{pred}_j = \sum_i W_{ij}\,\text{ref}_i$:

```{literalinclude} ../tools/line_model.py
:pyobject: moved_cells
```

```{literalinclude} ../tools/line_model.py
:pyobject: splat_weights
```

The voxel boundaries are smoothed. With sharp boundaries, a cell narrower than a voxel contributes
the same amount wherever it lies within that voxel. A small change of the field then leaves the
prediction unchanged, and the derivative with respect to the field is zero. unpile therefore
smooths each boundary with a triangular kernel of half-width 0.5 voxels (the `ramp` function).
The prediction then varies continuously with every edge position, and a gradient is available
everywhere.

## Comparison with resampling models

Most correction methods model the distortion in the opposite direction. They resample the acquired
image at the displaced positions to undo the distortion, and compare the result with an
undistorted image; this is a *pull* model. Registration-based fieldmap-less correction, such as
SyN-SDC, resamples only (Wang et al., 2017). TOPUP (Andersson et al., 2003) is also a pull model. It
resamples each acquired image and multiplies the result by the Jacobian of the displacement,
$1 + \partial d / \partial y$, which corrects the intensity change caused by compression and
stretching. It then compares the corrected images of the two phase-encoding directions with each
other. unpile uses a *push* model: values are carried forward from the undistorted image to the
acquired grid.

```{figure} ../_static/line_push_pull.svg
:alt: At the true field, the reference pushed forward against the acquired line, and the acquired line resampled with the Jacobian against the reference.

Both models evaluated at the true displacement, voxels 34–61. (A) Push: the reference
predicted onto the acquired grid, compared with the acquired b=0. (B) Pull: the acquired b=0
resampled and multiplied by the Jacobian, compared with the reference.
```

For a continuous image the two directions are equivalent; on a voxel grid they are not. In a
region of pile-up, one acquired voxel contains signal from about three reference voxels. Resampling
can only distribute that one value over the three positions, and the structure among them cannot
be recovered from the acquired image. The push model sums the reference, which still contains that
structure, onto the coarser acquired grid. It never needs to recover the lost detail.

The table gives the RMS difference at the true displacement, as a percentage of the median brain
signal. A perfect model would leave only noise; in the background of this image the noise standard
deviation is 4%.

| model | compared with | pile-up voxels (16) | stretched voxels (13) |
|---|---|---|---|
| push | acquired b=0 | 12% | 8% |
| pull × Jacobian | reference | 18% | 6% |
| pull without Jacobian | reference | 155% | 22% |

The push model is more accurate where signal piles up and slightly less accurate where it is
stretched. Pile-up regions are also where fieldmap-less correction most often fails, so the
difference matters most there. The last row shows why resampling alone performs poorly in these
regions: resampling cannot reduce the intensity of a voxel. A registration that only resamples can
reproduce a pile-up peak only by distorting the field.

## Objective

For a candidate field $f$, the prediction is compared with the acquired image. With a reference of
the same contrast (the clean b=0 here, or a synthetic b=0 from SynB0) the objective is

$$
L(f) = \frac{1}{|M|}\sum_{y \in M} \big(g\,\text{pred}_f(y) - \text{obs}(y)\big)^2
\; + \; \lambda\,\overline{\left(\partial^2 f\right)^2}
\; + \; \overline{\left(f/100\,\text{Hz}\right)^2}\Big|_{\text{outside the head}}
$$

- The first term is the mean squared difference over $M$, the brain mask dilated by four voxels.
  The intensity gain $g$ is a single scalar, solved in closed form at each evaluation, because the
  reference and the EPI are not on the same intensity scale. In this example the acquired b=0 is
  about 14% brighter than the reference.
- The second term is the bending energy, the mean squared second derivative of the field. It
  penalises rapid changes in the field. $\lambda$ is set with `--lam` (default 0.1) or `--lam-mm`.
- The third term draws the field towards zero more than three voxels outside the head, where there
  is no signal to determine it.

The sums run over the whole volume: one scalar objective covers every line, and all lines are fitted
together through the shared three-dimensional field. On this page the sums run over the one line.

With a T2w reference the first term is replaced by one minus the local normalized cross-correlation
in 5-voxel windows (`--metric lncc`). This compares the local shape of the two signals rather than
their intensities.

## The initial gradient

At zero field the prediction is the reference, scaled by the gain. Automatic differentiation
through the splat gives the derivative of the objective with respect to the field at every voxel.

```{figure} ../_static/line_gradient.svg
:alt: The prediction at zero field against the acquired line, and the descent direction for the field, per voxel and projected on the coarse B-spline grid, next to the true displacement.

(A) Prediction at zero field compared with the acquired b=0. (B) The direction in which the
objective decreases, expressed as a displacement: per voxel (orange), and after projection on the
24 mm B-spline grid used at the first level (blue). The true displacement is shown for comparison
(grey). Each curve is scaled to unit peak; only the shapes are compared.
```

The per-voxel derivative is local. It is large only near edges of the prediction that are
misplaced relative to the acquisition (voxels 42–48 and 63–69) and close to zero elsewhere. On its
own it would be a poor guide to the field.

The field, however, is not estimated voxel by voxel. It is represented as a cubic B-spline. At the
first level the control points are 24 mm (8 voxels) apart, and each basis function spans 32 voxels.
The derivative with respect to a control point is the sum of the per-voxel derivatives weighted by
its basis function, so the update is pooled over a wide neighbourhood. After this projection the
first update (blue) is smooth and already has the shape of the true displacement.

## Iterations

The B-spline coefficients are estimated with L-BFGS, a quasi-Newton method that approximates the
curvature of the objective from the gradients of previous iterations (Liu and Nocedal, 1989).
`src/lbfgs.rs` implements the same algorithm as `torch.optim.LBFGS` with a strong-Wolfe line
search, which is used on this page.

```{figure} ../_static/line_fit.svg
:alt: Animation of the prediction converging on the acquired line while the estimated displacement converges on the true displacement.

The complete estimation. Each frame is one iteration; later in each level, every sixth iteration
is shown.
```

```{figure} ../_static/line_fit_steps.svg
:alt: Snapshots of the prediction against the acquired line and of the estimated against the true displacement during the first level and at the end.

Snapshots at iterations 0, 1, 3, 10 and 60 of the 24 mm level and at the end of the 3 mm level.
Left: prediction (dashed) and acquired b=0. Right: estimated and true displacement.
```

| level | iteration | RMS displacement error in the brain |
|---|---|---|
| 24 mm | 0 | 9.2 mm |
| 24 mm | 1 | 7.8 mm |
| 24 mm | 3 | 2.6 mm |
| 24 mm | 10 | 1.3 mm |
| 24 mm | 60 | 1.1 mm |

By the third iteration the pile-up peak has formed in the prediction, and by the tenth it is in the
correct position. The remaining error at the end of the level reflects the limited flexibility of a
24 mm control grid, which cannot follow the sharper changes of the true field.

## Coarse-to-fine schedule

The default schedule (`--levels 24,12,6,3`) estimates the field on control grids of decreasing
spacing. Each level starts from the field of the previous one. At every level except the last,
the prediction and the acquired image are both smoothed with a Gaussian kernel whose standard
deviation is half the control-point spacing, limited to 3 voxels.

Smoothing enlarges the range of fields from which the estimation converges. If the field is wrong by
several millimetres, sharp edges in the prediction and the acquisition do not overlap at all, and
the derivative carries no information about which way to move them. After smoothing, the same
error produces a broad intensity mismatch whose gradient points towards the solution. In the same
way, two blurred photographs can be aligned roughly before their details are compared.

```{figure} ../_static/line_levels.svg
:alt: The data term and bending energy over all iterations of all levels, and the estimated displacement after each level against the truth.

(A) Data term and weighted bending energy over the full schedule. The data term increases at the
start of each level because the smoothing is reduced, which makes the comparison stricter. (B)
Estimated displacement at the end of each level.
```

| control-point spacing | smoothing (standard deviation) | RMS displacement error in the brain |
|---|---|---|
| 24 mm | 9 mm | 1.12 mm |
| 12 mm | 6 mm | 0.53 mm |
| 6 mm | 3 mm | 0.36 mm |
| 3 mm | none | 0.44 mm |

The error increases slightly at the last level. Without smoothing, the 3 mm grid fits part of the
noise along this line. In the volume, the bending energy couples each line to its neighbours in the
other two directions, which limits this effect.

## Corrected image

With the field estimated, the corrected image is the image that, pushed forward, best explains the
acquisition. unpile computes it as the least-squares solution with a small smoothness penalty,
using conjugate gradients (`unwarp` in `src/problem.rs`), and writes it to `corrected.nii.gz`.

```{figure} ../_static/line_unwarp.svg
:alt: Corrected lines from the estimated field: resampled, resampled with the Jacobian, and the least-squares inverse of the push, against the reference.

Corrected line at the estimated field, voxels 34–61, divided by the fitted gain so that it is on the
reference's intensity scale.
```

The displacement field `sdc_warp.nii.gz` is applied by `antsApplyTransforms` as a resampling
(dotted). It returns signal to the correct position but does not change intensities, so the pile-up
remains as a broad bright band. Multiplying by the Jacobian corrects the intensities (dashed). The
least-squares inverse of the push also recovers the structure within the pile-up. In the pile-up
voxels, the RMS difference from the reference is 152% for resampling, 16% for resampling with the
Jacobian, and 9% for the inverse of the push.

## From one line to the volume

The volumetric estimation minimises the same objective with the following differences:

- The field is a single three-dimensional cubic B-spline. Each line along the phase-encoding axis is
  pushed forward separately, but neighbouring lines share control points, and the bending energy
  includes second derivatives along all three axes and the mixed derivatives. Neighbouring lines
  therefore constrain each other.
- With a T2w reference (`--metric lncc --pose`), a rigid transformation of the T2w is estimated
  together with the field. A uniform shift along the phase-encoding axis can be explained either as
  a constant field or as a translation, so the mean of the field inside the brain is penalised
  towards zero.
- With a gradient-echo fieldmap (`--fieldmap`), the field is the fieldmap under its own rigid
  transformation, plus a second-order shim polynomial, plus a B-spline residual with a ridge
  penalty. The same splat and data term are used to estimate all three; see {doc}`gre-refinement`.
- With a pair of opposite phase-encoding directions (`--reverse-epi`) there is no reference. The
  undistorted image is re-estimated between rounds as the image that, pushed forward with $+d$ and
  $-d$, best explains both acquisitions; see {doc}`opposite-polarity`.

## Reproducing the figures

```bash
pip install torch nibabel matplotlib
python docs/tools/make_line_walkthrough.py --testdata testdata/rpe --out docs/_static
```

The script first checks the line operators against the prototype outputs, then runs the
estimation and prints the values quoted on this page. It differs from `unpile fit` in the following
respects: the bending energy and the smoothing act along the line only; the corrected line is
obtained with a direct solver instead of 25 conjugate-gradient iterations; and $\lambda$ is the
command-line default, which was not tuned for one dimension. The estimation uses the triangular
boundary kernel of `unpile fit`. The operator check uses the prototype's Gaussian kernel, with which
the stored outputs were made.

## References

- Andersson, J. L. R., Skare, S., and Ashburner, J. (2003). How to correct susceptibility
  distortions in spin-echo echo-planar images: application to diffusion tensor imaging.
  *NeuroImage*, 20(2), 870–888.
- Liu, D. C., and Nocedal, J. (1989). On the limited memory BFGS method for large scale
  optimization. *Mathematical Programming*, 45, 503–528.
- Wang, S., Peterson, D. J., Gatenby, J. C., Li, W., Grabowski, T. J., and Madhyastha, T. M.
  (2017). Evaluation of field map and nonlinear registration methods for correction of
  susceptibility artifacts in diffusion MRI. *Frontiers in Neuroinformatics*, 11, 17.
