# Validation

Every number in these pages rests on one of two kinds of evidence, and the pages say which.

## Simulated data with ground truth

[TRXScan](https://github.com/PennLINC/TRXScan) simulates a diffusion acquisition of a real
subject's anatomy with a known field and known head movements. It writes the truth next to the data:
the field, the undistorted b=0, and the rigid transforms it applied between the EPI and the
anatomical or fieldmap scans. The book's [artifact-free reference](https://pennlinc.github.io/diffusion-book/book/preprocessing/susceptibility-distortion#the-artifact-free-reference) is
this undistorted b=0, and its chapter builds the same kind of distortion with the same simulator.

unpile's integration tests run on two such datasets, generated from the recipes in
`tests/fixtures/trxscan_fixtures.txt`, and score every fit against the truth. The assertions as the
test prints them:

```text
phasediff GRE prior: field corr = 0.998 within [0.98, 1]
phasediff GRE prior: pose residual rotation deg = 0.770 within [0, 3]
phasediff GRE prior: pose residual translation mm = 0.581 within [0, 1.5]
phasediff T2w: pose residual rotation deg = 1.302 within [0, 3]
phasediff T2w: pose residual translation mm = 2.038 within [0, 3]
rpe two-blip: field corr = 0.987 within [0.97, 1]
```

The simulated data has limits that matter when reading these numbers. Its b=0 is a tissue-class
image without the texture of a real one, it has no skull, and its fieldmap noise is a little higher
than a scanner's. A result that looks better on the simulation than on real data usually comes from
one of these.

## Real data without truth

A small bundle of derived OpenNeuro data (CC0 and CC-BY, about 40 MB) is published as a GitHub
release and fetched by `scripts/fetch_testdata.sh`: a 3 mm GE-EPI series with a measured fieldmap and
a T1w (ds001600), a 2.4 mm single-band reference with a spin-echo pair, a T1w and a T2w on a 23°
oblique grid (ds002737), and two sessions of spin-echo and gradient-echo pairs with a T2w (BOLD5000).
Without truth, the tests on these check agreement with the measured fieldmap, consistency between
contrasts, and regression bounds on recorded values (`tests/fixtures/bounds.toml`).

## Operator check

`unpile check` compares each operator of the forward model with dumps from the PyTorch prototype
the Rust code was ported from. CI runs it on every push; the agreement is at the 1e-6 level.
