# Configuration guide

Every workflow script takes one argument, the YAML configuration file.
`workflows/config_template.yaml` lists every option with its default
value; a user file only needs the options that differ from the defaults.
Paths are relative to the folder of the configuration file. Parameter
values are always written in **physical units**; internally the
framework standardises the inputs (`x_s = (x - mean) / std`, fitted on
all training runs).

## Parameters

```yaml
parameters:
  theta_1: {calibrate: true,  bounds: [0.5, 1.5]}
  theta_2: {calibrate: false, fixed_value: 1.0}
```

* one entry per parameter column of `inputs` in `input_data.h5`, same name;
* `calibrate: false` keeps the parameter at `fixed_value` in every GP
  evaluation (the GPs are always trained on all parameters);
* `bounds` are only used to build the Morris design (step 01). The prior
  bounds of the calibration are the range of the training runs.

### Fixed values that change from case to case

`data.fixed_values_file` points to a CSV with a column `case` and one
column per non-calibrated parameter whose value is known but different
in every experiment (for instance the true values of a synthetic study):

```
case,theta_2,theta_3
case_01,0.93,1.02
case_02,1.05,0.97
```

For the listed cases these values replace `fixed_value` in every GP
evaluation (likelihood, predictions, validation); the other cases, and
the samples exported for a new case, use `fixed_value`.

## Cases for calibration and validation

`data.validation_cases` is a list of case ids, or the path of a CSV with
columns `case` and `set` (rows with `set == validation`). Step 02 writes
such a file, `results/correlation/validation_split.csv`, when
`sensitivity.correlation.n_validation` > 0: for every case the largest
squared Spearman coefficient rho^2 (as for sensitivity indices, no
absolute value needed) with the other cases among the pairs that remain
significant after the Benjamini-Hochberg correction; the `n_validation`
cases with the lowest value (the least correlated with the others) are
kept for validation.

The tables of step 02 (`input_output_summary.csv`) help decide which
parameters to calibrate.

## Measurement uncertainty

Relative: `sigma = relative * |measured value|`, with optional per-case
values. For a time-dependent QoI the covariance `diag(sigma^2)` is
projected on the retained principal components, `W diag(sigma^2) W^T`.

## Surrogates

`surrogate.file` sets where the GPs are saved and read; give the same
file in several configurations to share one set of GPs (e.g. several
calibration studies on the same simulations with different measurements).

Kernel comparison: `python workflows/03_train_surrogates.py config.yaml
--compare-kernels` trains every entry of `surrogate.kernel_candidates`
on the same train/test split and reports, on the test runs, the relative
RMSE, Q2, mean and standard deviation of the standardised residuals
(ideally 0 and 1), the coverage of the 95% interval and the negative log
predictive density (NLPD, lower is better). The recommended kernel has
the lowest NLPD among those whose standardised residuals have a standard
deviation in [0.8, 1.25]. `kernel_comparison_cases` restricts the
comparison to some cases. Every kernel is `C x (terms)` [+ white noise],
with one length scale per parameter for RBF and Matern.

## Likelihood

`likelihood.include_gp_variance: true` adds the GP predictive variance
to the measurement covariance, in both the standard and the
hierarchical calibration, so that the two methods stay comparable.

## Standard calibration (emcee)

* `stretch_a`: scale of the stretch move (Goodman & Weare, 2010). The
  emcee default is 2; a larger value gives larger moves and a lower
  acceptance rate.
* Tuning: step 1 tries each value of `walker_multipliers`
  (`nwalkers = multiplier x number of calibrated parameters`, at least
  2 x that number) for `nsteps_tuning` steps, step 2 tries each value of
  `nsteps_list`. With a single value in a list, that step is skipped.
  With `stop_at_first_pass: true` the lengths are tried in increasing
  order and step 2 stops at the first one that meets the convergence
  criteria; the acceptance is then checked at that length.

## Hierarchical calibration (MH-within-Gibbs)

* `burnin_fraction`: the proposal adaptation stops at the end of the
  burn-in, so the samples kept are from a fixed (valid) Markov kernel.
* `adaptation.mu`, `adaptation.sigma`, `adaptation.X`: adaptation of each
  type of block, `acceptance_rate` or `covariance`:
  * `acceptance_rate`: the proposal scale is multiplied by
    `exp(0.01 (AR - target))` at each step, from `adaptation.start` to the
    end of the burn-in; targets in `adaptation.target_acceptance`
    (0.44 for the 1D blocks mu_d and sigma_d, 0.234 for X with several
    parameters);
  * `covariance`: adaptive Metropolis (Haario et al., 2001). From
    `adaptation.start` the empirical covariance of the block is
    accumulated, and from `proposal_update_start` the proposal
    covariance becomes `s_d (C + epsilon I)`. `s_d: null` uses
    `2.4^2 / d`; `s_d` can also be given per block, e.g.
    `s_d: {mu: 3.0, sigma: 3.0, X: 1.5}` (a lower value gives a higher
    acceptance rate). For mu_d and sigma_d this estimates the covariance of the
    marginal chain, wider than the conditional one used in a Gibbs step,
    which is why `acceptance_rate` is the default for these blocks.
* `nsteps_list`: lengths tried in order until the convergence criteria are
  met (only the final run is saved; `HC_<time>_lengths.csv` lists all).
* `adaptation.stop_before_burnin_end`: the adaptation (and the step size of
  the joint shift move) stops this many iterations before the end of the
  burn-in, so the end of the burn-in already runs with the final proposal.
* `parallel_chains: true`: the chains run at the same time, one per CPU
  core (the progress bars are then hidden).
* `initial_proposal_variance`: starting proposal variances (standardised
  units) for the `mu`, `sigma` and `X` blocks.
* `dispersed_init`: chain 1 starts at the centre of the prior range,
  chains 2..N at random points.

### `fixed_proposal_cov`

With covariance adaptation, the proposal of a case can collapse: the
empirical covariance shrinks, the acceptance rate goes to zero and the
chain stops moving. Step 06 detects it and prints a covariance estimated
in the window before the collapse. Copy it into the configuration: for
that case the proposal of `X` becomes `N(x_current, Sigma)` with this
fixed `Sigma` and no adaptation; the other cases adapt normally.

Example, two calibrated parameters (rows/columns in the order of the
parameters with `calibrate: true`), standardised units:

```yaml
hierarchical:
  fixed_proposal_cov:
    case_07:
      - [ 3.349e-6, -3.174e-6]
      - [-3.174e-6,  3.033e-6]
```

### Joint shift move

`joint_shift_move.enabled: true` adds, after every Gibbs sweep and for
each parameter d, a move that shifts `mu_d` and `X[:, d]` of all cases by
the same random amount. The hierarchical term does not change, so the
acceptance only depends on the likelihood: `mu` can move even when
`sigma` is small, which otherwise slows the mixing of `mu`. The step size
is adapted towards `target_acceptance` during the burn-in only.

### Joint scale move

`joint_scale_move.enabled: true` adds, after the shift move and for each
parameter d, a move that multiplies `sigma_d` and the deviations
`X[:, d] - mu_d` of all cases by the same factor `lam = exp(eps)`,
`eps ~ N(0, s^2)` (scale group move of Liu & Sabatti, 2000,
Biometrika 87(2) 353-369, doi:10.1093/biomet/87.2.353). With the uniform
prior on `sigma`, the hierarchical term (ratio `lam^-E`) and the Jacobian of
the `E + 1` scaled coordinates (`lam^(E+1)`) give the log acceptance
`dloglik + eps`. When the population std is close to zero (no inter-case
variability, e.g. a purely epistemic case) the case values X cannot move
away from `mu` unless `sigma` grows, and `sigma` cannot grow unless the X
spread (funnel of hierarchical models): the Gibbs updates then mix very
slowly and the X acceptance drops whenever `sigma` is small. The scale move
updates `sigma` and the X together and removes this coupling. Its step `s`
is adapted towards `target_acceptance` during the burn-in only.

## Diagnostics

Criteria: R-hat <= `rhat_max`; bulk ESS >= `ess_per_chain` x number of
chains (or walkers); tail-ESS too if `check_tail_ess` (hierarchical);
N >= `tau_factor` x tau if `tau_factor` is set (N = post-burn-in draws
per chain, tau = integrated autocorrelation time: emcee's estimate for
the standard calibration, N_total / bulk-ESS otherwise); acceptance rate
inside the range of the block (`acceptance_range_standard` for emcee,
`acceptance_range_1d` for mu_d and sigma_d, `acceptance_range_X` for X
with several parameters); MCSE/sigma by overlapping batch means, checked
only if `mcse_over_std_max` is set. The MCSE divides by the draws of one
chain (conservative choice).

For emcee, `standard_rhat: false` drops R-hat (the walkers of an ensemble
are not independent chains) and `standard_ess: autocorr` computes the ESS
as N x n_walkers / tau. `hierarchical_convergence_blocks` lists the
blocks that must converge (e.g. `[mu, sigma]`); the others are reported,
and the acceptance of every block is always checked. The tables give two
verdicts: `converged` (decides the length of the chains) and `pass`
(converged and acceptance in range).

The hierarchical table has one row per block: mu_d, sigma_d and X[case, d].
The acceptance of each block is recorded by its own sampler at every
step, so it is not affected by the joint shift move (whose acceptance is
reported separately). Every failed criterion is followed by a
SUGGESTION, e.g. which option to change and by how much.
