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

The tables of step 02 (`input_output_summary.csv`) help decide which
parameters to calibrate.

## Measurement uncertainty

Relative: `sigma = relative * |measured value|`, with optional per-case
values. For a time-dependent QoI the covariance `diag(sigma^2)` is
projected on the retained principal components, `W diag(sigma^2) W^T`.

## Likelihood

`likelihood.include_gp_variance: true` adds the GP predictive variance
to the measurement covariance, in both the standard and the
hierarchical calibration, so that the two methods stay comparable.

## Standard calibration (emcee)

* `stretch_a`: scale of the stretch move (Goodman & Weare, 2010). The
  emcee default is 2; a larger value gives larger moves and a lower
  acceptance rate.
* Tuning: step 1 tries each value of `walker_multipliers`
  (`nwalkers = multiplier x number of calibrated parameters`) for
  `nsteps_tuning` steps, step 2 tries each value of `nsteps_list`. With a
  single value in a list, that step is skipped.

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
    `2.4^2 / d`. For mu_d and sigma_d this estimates the covariance of the
    marginal chain, wider than the conditional one used in a Gibbs step,
    which is why `acceptance_rate` is the default for these blocks.
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

## Diagnostics

Criteria: R-hat < `rhat_max`; bulk ESS >= `ess_per_chain` x number of
chains (or walkers); acceptance rate inside the range of the block
(`acceptance_range_standard` for emcee, `acceptance_range_1d` for mu_d
and sigma_d, `acceptance_range_X` for X with several parameters);
MCSE/sigma by overlapping batch means, checked only if
`mcse_over_std_max` is set. The MCSE divides by the draws of one chain
(conservative choice).

The hierarchical table has one row per block: mu_d, sigma_d and X[case, d].
The acceptance of each block is recorded by its own sampler at every
step, so it is not affected by the joint shift move (whose acceptance is
reported separately). Every failed criterion is followed by a
SUGGESTION, e.g. which option to change and by how much.
