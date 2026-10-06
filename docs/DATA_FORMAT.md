# Data format specification

This document defines the files a user must provide to run the
Advanced Bayesian Calibration Framework. How the data are produced
(Dakota, a custom Python driver, another sampling tool, ...) is up to
the user: the framework only requires that the files follow the
structure below. Every file is checked by a validator before any
computation starts, and any deviation is reported with an explicit
error message.

The same format is used for **scalar** and **time-dependent**
quantities of interest (QoIs): a scalar QoI is a time-dependent QoI
with a single time point (`n_t = 1`).

Notation:

| Symbol    | Meaning                                              |
|-----------|------------------------------------------------------|
| `p`       | number of model parameters                           |
| `n_eval`  | number of solver evaluations used to train the GP    |
| `n_t`     | number of time points of the QoI (`1` for scalar)    |
| `n_post`  | number of posterior samples used for validation      |

---

## 1. Calibration input — `input_data.h5` (required)

```
input_data.h5
├── inputs                      structured table, one row per solver evaluation
│     eval_id   int             unique identifier of the evaluation
│     <param_1> float           value of parameter 1
│     ...
│     <param_p> float           value of parameter p
│
└── cases/                      one group per experiment
      └── <case_id>/
            tim       (n_t,)          float   time grid ([0.0] for a scalar QoI)
            meas      (n_t,)          float   measured QoI
            pred      (n_eval, n_t)   float   simulated QoI, one row per evaluation
            eval_ids  (n_eval,)       int     eval_id of each row of `pred`
            filter    (n_t,)          bool    OPTIONAL — points to use (default: all True)
```

Rules:

- Parameter names in `inputs` must match the names used in `config.yaml`.
- Each `eval_ids` entry must exist in `inputs/eval_id`. Failed solver
  runs are simply omitted from `pred` and `eval_ids`; `n_eval` may
  therefore differ between cases.
- `meas`, `tim` and `filter` must have the same length `n_t`; `pred`
  must have `n_t` columns.
- `<case_id>` is any string without `/`.
- The framework is solver-agnostic: it makes no assumption on the
  physical model, the simulation code or the tool used to sample the
  parameters. Any model that maps `p` parameters to a QoI can be used.

A helper script, `tools/csv_to_input_h5.py`, builds this file from
CSV tables (one CSV of training samples per case plus one CSV of
measurements), which is the most common situation for scalar QoIs.

## 2. Synthetic reference — `synthetic_truth.h5` (optional)

Required only when `synthetic: true` in `config.yaml`. The true
values are overlaid on the posterior comparison plots.

```
synthetic_truth.h5
└── cases/
      └── <case_id>/
            theta  (p,)  float   true parameter values of this case (all parameters)
```

The plots mark the true values of the calibration cases (filled) and of
the validation cases (open). With
`postprocessing.show_reference_population: true` they also draw the
normal distribution with the mean and standard deviation of the true
values of the calibration cases.

Parameter order follows `config.yaml`. Case identifiers must match
those in `input_data.h5`.

## 3. Validation input — `validation_<method>.h5` (optional)

Used by `08_validation.py` when `run_validation: true`. The user
re-runs the solver with posterior samples from a calibration and
stores the predictions here. `<method>` is `SC` (standard
calibration) or `HC` (hierarchical calibration); one file per method.

```
validation_<method>.h5
└── cases/
      └── <case_id>/
            tim     (n_t,)           float   time grid
            meas    (n_t,)           float   measured QoI
            pred    (n_post, n_t)    float   solver predictions, one row per
                                             posterior sample
            theta   (n_post, p)      float   OPTIONAL — posterior samples used
```

The posterior samples to run can be exported with
`python workflows/08_validation.py config.yaml --export-samples 200`
(files `results/validation/posterior_samples_SC.csv` and `_HC.csv`; for
the hierarchical calibration they are draws for a new case from the
population distribution).

Validation cases may differ from calibration cases (for example,
experiments held out from the calibration).

For each case and method, `08_validation.py` computes:

- **z-score** of the measurement with respect to the posterior
  predictive distribution, at each time point;
- **coverage**: fraction of measured points falling inside the
  central predictive interval (level set in `config.yaml`, default 95%);
- **RMSE** between the posterior predictive mean and the measurement.

When both `validation_SC.h5` and `validation_HC.h5` are present, the
two methods are compared on the cases they share.

## 4. Measurement uncertainty — `config.yaml`

The measurement uncertainty is not stored in the data files. It is
set for every calibration run in `config.yaml`, because it depends on
the study:

The uncertainty is **relative**: the standard deviation of each
measured point is a fraction of the measured value,
`sigma = relative_noise * |meas|`.

```yaml
measurement_noise:
  relative: 0.01          # 1% of the measured value, applied to every case
  per_case:               # OPTIONAL overrides
    case_03: 0.08
```

In the standard calibration this value enters the Gaussian likelihood
evaluated by `emcee`. In the hierarchical calibration it is combined
with the GP predictive uncertainty in the likelihood of the local
(case-level) variables.

## 5. Results files (written by the framework)

`results/calibration/SC_<timestamp>.h5` and `HC_<timestamp>.h5`:

```
metadata   (attributes) method, qoi_type, case_ids, param_names, burnin, nsteps,
           seeds, wall_clock_s, include_gp_variance, relative_noise, ...
scaler/    mean, std, lower, upper   (standardisation of the inputs)
chains/    SC: theta (n_walkers, n_steps, p), log_prob, acceptance
           HC: mu, sigma (n_chains, n_steps, p), X (n_chains, n_steps, n_cases, p), logpost
```

Chains are stored in standardised units; physical values are
`x * std + mean` (and `sigma * std` for the population standard
deviation). Existing results are never overwritten.
