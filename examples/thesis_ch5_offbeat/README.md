# Thesis, Chapter 5: scalar case studies with OFFBEAT

Standard (emcee) and hierarchical (MH-within-Gibbs) calibration of the
fuel centreline temperature at end of life (FCT at EOL) of 130 fuel rods
derived from Turkey Point 3 (TP3), simulated with OFFBEAT, in three synthetic
situations where the true parameters are known:

| Case | Data | rho, relocation, swelling in the GPs | Likelihood of rod *i* |
|---|---|---|---|
| epistemic | dataset A (one true vector for all rods) | true values (common) | N(y_i \| GP_i(k, alphaT, theta_true)) |
| aleatoric | dataset B (one true vector per rod) | true values of rod *i* | N(y_i \| GP_i(k, alphaT, theta_true,i)) |
| modelbias_uniform_k | dataset A | alphaT, rho, relocation, swelling at the OFFBEAT default 1.0 for every rod (bias common to all rods); only k calibrated | N(y_i \| GP_i(k, 1, 1, 1, 1)) |
| modelbias_struct_k | dataset A | alphaT at 1.0; rho, relocation, swelling linear in the time-averaged LHGR of rod *i* (`make_modelbias_struct.py`): bias dependent on the operating conditions; only k calibrated | N(y_i \| GP_i(k, 1, theta(P_i))) |

The 130 rods share the Turkey Point 3 geometry and coolant temperature
and use the power history (linear heat rate, axial profile, fast flux)
of their TP3 rod. The same rods are used for the training runs and for
the synthetic experiments; the stage `gp` checks it (GP at the true
values vs OFFBEAT, file
`results/common/surrogates/gp_vs_offbeat_at_true_values.csv`).

Calibrated parameters: UO2 thermal conductivity `UO2_k` and thermal
expansion `UO2_alphaT` (multipliers of the nominal value), the two
parameters of the GPs with a total Sobol' index ST >= 0.01 under the prior
uncertainty (stage `sobol`); density, relocation and swelling are fixed. Measurement uncertainty: 1%
of the measured value, in the likelihood and in the validation. The GP
predictive variance is included in the likelihood of both methods.

## Data (`inputs/`)

| File | Content |
|---|---|
| `lhs_inputs.csv` | 600 OFFBEAT runs: Dakota LHS of the 5 multipliers in [0.5, 1.5] (nominal 1), seed 4344 |
| `lhs_FCT_EOL.csv` | FCT at EOL [K] of the 600 runs, one column per rod (the same parameter samples for every rod). Empty cells: 314 of the 78 000 rod runs (0.4%, at most 13 per rod) stopped before the end of life, all at low conductivity, high thermal expansion and high relocation (the hottest corner of the domain); they are excluded for that rod only |
| `synthetic_experiments.csv` | synthetic experiments: true parameters of every rod for datasets A and B, OFFBEAT FCT at EOL without (`FCT_EOL_model_K`) and with 1% Gaussian noise (`FCT_EOL_exp_K`) |
| `metadata.json` | seed, distributions of the true values, true vector A, values of the model-bias case |
| `morris_tabular.dat` | Dakota tabular of the Morris screening (psuade_moat, 17 OFFBEAT multipliers in [0.5, 1.5], 10 trajectories x 18 = 180 evaluations, 4 levels, seed 500), FCT at EOL of the 130 rods; empty = run stopped before the end of life (1360 of the 23 400 rod runs, 5.8%; 91.8% of the elementary effects available) |

Distributions of the true values (multipliers of the OFFBEAT default). The
widths come from the literature; the centres are moved away from the
default (1.0), because the reality does not coincide with the nominal
model, and away from the region where OFFBEAT does not reach the end of
life (low k, high alphaT): k ~ N(1.15, 0.10^2), alphaT ~ N(0.90,
0.075^2), swelling ~ N(1.15, 0.15^2) truncated to [0.5, 1.5], relocation
~ U(0.85, 0.95) (widths: Fiorina et al. 2022, Table 1, from Che et al.
2018); density ~ N(0.9874, 0.0086^2), i.e. (93.8 +- 1.6)% TD at the
2.5/97.5 percentiles (Ikonen 2016, Table 2,
doi:10.1016/j.nucengdes.2015.11.025) divided by the OFFBEAT default
density fraction 0.95. The synthetic experiments were produced by
OFFBEAT with these true values, rounded to 6 significant digits as
written in `system/userParameters` (script
`synthetic_rods.py`, not part of the framework).

## Sampler settings

Tuned on the quick run (10 calibration rods, 10 000 steps):

* emcee: `stretch_a: 4.0` (3.0 gave acceptance 0.52-0.59; 4.0 gives 0.44).
* MH-within-Gibbs: `s_d: {mu: 0.5, sigma: 1.0, X: 0.7}` instead of 2.4^2/d,
  and `joint_scale_move` enabled. With the default settings sigma did not mix
  (bulk-ESS 10-40, tau 600-1900): in the epistemic and model-bias cases the
  population std is close to 0, the X cannot leave mu unless sigma grows and
  sigma cannot grow unless the X spread (funnel of hierarchical models; the X
  acceptance drops to ~0 whenever sigma is small). The scale move (sigma and
  X - mu multiplied by the same factor; Liu & Sabatti 2000,
  doi:10.1093/biomet/87.2.353) gives a sigma bulk-ESS of 700-1270 at 10 000
  steps (epistemic and aleatoric). Checked against the exact posterior of a
  linear-Gaussian hierarchical model (same distribution, ESS of sigma x48).

## Run

From this folder, with the `abcf` environment active:

```bash
python run_thesis_ch5.py all --quick     # pipeline check, a few minutes (16 rods, short chains)
python run_thesis_ch5.py                 # thesis results (several hours)
```

Before running, the old outputs of the selected stages are deleted, so that a new run never
mixes with files of a previous run: with `all` (or `prepare`) the whole `data/` and
`results/` (`results_quick/`) folders, otherwise only the folders of the selected stages
(e.g. `epistemic` deletes only `results/epistemic/`). Add `--keep` to keep them.

or stage by stage:

```bash
python run_thesis_ch5.py prepare correlation morris kernels gp sobol   # common part (GPs ~1-2 h)
python run_thesis_ch5.py epistemic                         # one case (also aleatoric, modelbias_uniform_k, modelbias_struct_k, epistemic_prior)
python run_thesis_ch5.py figures                           # figures and tables
```

| Stage | What it does | Output |
|---|---|---|
| prepare | CSV -> `data/*.h5` | `data/` |
| correlation | Spearman between the FCT of every pair of rods on the 600 runs (common valid runs; two-sided tests, Benjamini-Hochberg); for each rod the largest significant rho^2 (squared coefficient, as for sensitivity indices); the 6 rods with the lowest value -> validation, the other 124 -> calibration | `results/common/correlation/validation_split.csv` |
| morris | elementary effects per rod (consecutive points of each trajectory, only if both runs reached the end of life), mu* and sigma per rod, mean and 2.5-97.5 percentiles across the rods; vertical line: 1% of the mean FCT (measurement uncertainty) | `results/common/morris/` |
| kernels | 5 kernels (C x RBF, C x Matern 5/2, with and without white noise, C x (Matern 3/2 + RQ) + white noise), same 80/20 split: relative RMSE, Q2, standardised residuals, 95% coverage, NLPD | `results/common/surrogates/kernel_comparison_*` |
| gp | one GP per rod on the 5 parameters, kernel of `config_surrogates.yaml` | `results/common/surrogates/` |
| sobol | Sobol' indices S1, ST of the 5 GP parameters on the GP mean of every rod, with the prior uncertainty (literature widths, centred on the OFFBEAT default, truncated to [0.5, 1.5]); ST >= 0.01 -> worth calibrating, below -> can be fixed | `results/common/sobol/` |
| epistemic, aleatoric, modelbias_uniform_k, modelbias_struct_k | steps 04-08 of the framework + 93 posterior samples (Wilks 95/95 two-sided; the first 59 are enough for one-sided) to re-run OFFBEAT | `results/<case>/` |
| epistemic_prior | the epistemic case again with literature normal priors on k and alphaT (N(1, 0.10^2), N(1, 0.075^2), Fiorina et al. 2022; standard: on theta, hierarchical: on mu), to show which results come from the data and which from the prior | `results/epistemic_prior/`, figure `prior_sensitivity_epistemic` |
| figures | parameter comparison, traces, posterior predictive; summary tables | `results/figures/` |

### Continuing chains instead of restarting them

With `continue_chains: true` (default, standard and hierarchical blocks) a longer length of
`nsteps_list` continues the chains of the previous length (10k -> 30k runs 20k new steps).
Hierarchical: the proposals adapted during the burn-in of the first length are frozen and
the random-number state is kept, so the continued chain is the same Markov chain run longer;
the burn-in is always `burnin_fraction` x the total length. emcee (no adaptation): a
continued ensemble is identical to one run at once (`tests/test_core.py`). Since this
version the HC files also store the chain checkpoints, so a converged-or-not run can be
continued later:

```bash
python run_thesis_ch5.py modelbias_struct_k --n-cal 16 --hc-steps 100000 --continue-hc
```

(`--continue-hc` keeps the SC result and continues the last HC file of the case; files written
by older versions have no checkpoints and cannot be continued.)

### Model bias: repetitions and figures (`repeat_ch5.py`, `figures_bias_ch5.py`)

The two model-bias cases calibrate k only (alphaT, not identifiable from the FCT at EOL, is
fixed at the default) and differ only in the dependence of the bias on the rod power. Each is
repeated on 100 synthetic data sets (same rods, new 1% measurement noise), at the chain
lengths of its converged reference run (not checked again): coverage of the true k by the 95%
intervals of SC, of the HC population mean and of the HC population predictive
(`results/repeat_<case>/coverage.csv`, `repetitions.csv`). Each repetition uses 4 cores.

```bash
python run_thesis_ch5.py modelbias_uniform_k modelbias_struct_k --n-cal 16 --hc-steps 10000 30000 50000
python repeat_ch5.py modelbias_uniform_k --n-cal 16 --reps 100 --jobs 14
python repeat_ch5.py modelbias_struct_k --n-cal 16 --reps 100 --jobs 14
python figures_bias_ch5.py       # results/figures/bias_coverage_ch5, bias_rodlevel_ch5
```

Kernel of the GPs: if the stage `kernels` has been run, the stage `gp`
uses the recommended kernel (lowest NLPD among the kernels whose
standardised residuals have a standard deviation in [0.8, 1.25]) and
prints it; otherwise `surrogate.kernel` of `config_surrogates.yaml`.

## Criteria (calibration_validation_criteria.xlsx)

| | Hierarchical (MH-within-Gibbs) | Standard (emcee) |
|---|---|---|
| chains | 4 independent chains, dispersed starts | 4 walkers |
| length | 10k -> 30k -> 50k -> 100k -> 150k until converged, burn-in 50% | same |
| adaptation | Haario on every block from iteration 200 (proposal updated from 400) to burn-in - 200, then fixed | none (stretch move, `stretch_a`) |
| convergence | mu, sigma: R-hat <= 1.01, bulk- and tail-ESS >= 400, N >= 50 tau | ESS = N * 4 / tau >= 400, N >= 50 tau |
| reported only | X of every rod: worst R-hat, minimum ESS; MCSE; wall clock | MCSE; wall clock |
| acceptance | mu, sigma [0.31, 0.57]; X [0.23, 0.48] | [0.2, 0.5] |
| validation | chi2_E and chi2_tot on the 6 validation rods, 4000 draws (HC: new-rod population) | same |

If an acceptance rate is out of range, step 06 prints a suggestion:
change `hierarchical.adaptation.s_d` (Haario scaling) or
`standard.stretch_a` and run the case again.
