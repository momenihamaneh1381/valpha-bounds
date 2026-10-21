# BLR experiment protocol

Run commands and dependency setup are in [README.md](README.md). All data are
generated synthetically; the seed and complete budgets are stored in
`outputs/blr_conditional_split_conditional_bounds_extended_config.json`.

## Parameters and selection

The study uses dimensions `1, 2, 5, 10`, sample size `n=100`, clipped squared
loss in `[0,4]`, noise variance `0.16`, prior precision `2`, true coefficient
`0.8`, and RNG seed `20260419`. The alpha grid is `1.05, 1.10, ..., 3.00`.

The conditional experiment uses 20 supersamples: 10 select alpha and 10
independently evaluate the selected value. Each supersample uses 24 selector
samples, 32 marginal components, and 384 weight samples. Parameter-space
alpha selection and evaluation each use four independent repeats. The
finite-candidate Theorem 1 curve uses 64 posterior means, 128 marginal
samples, 32 prior samples, and two anchors. Finite-candidate sensitivity
maxima are diagnostics rather than certified global suprema.

## MI, CMI, and uncertainty

Information is measured in nats. For loss in `[0,c]`, the baselines are:

```text
MI:   c * sqrt(I(W;S) / (2*n))
CMI:  c * sqrt(2 * I(W;U | supersample) / n)
```

MI uses the single-example subgaussian proxy `c**2/4`; CMI uses the
supersample loss-difference proxy `c**2`.

The MI curve averages four independently estimated bound values, with SEM
across those values. `mi_estimate` is the mean information estimate, so its
square-root transform need not equal the mean bound. For four repeats:

```text
(c**2/(2*n))*mi_estimate = mi_bound**2 + 3*mi_bound_stderr**2
```

CMI transforms the mean supersample information estimate and uses a
delta-method standard error. Alpha-dependent curves report SEM only from
the independent evaluation pool, at the alpha chosen on the selection pool.
Empirical generalization reports SEM across datasets.

## Saved outputs

`outputs/` contains the summary CSV, alpha-specific CSV, sanity checks,
Theorem 1 proxy summary, configuration, and the two figure PDFs/PNGs.
The summary CSV drives both paper figures; the other tables retain the
underlying diagnostics. A redraw does not rerun Monte Carlo estimation.
