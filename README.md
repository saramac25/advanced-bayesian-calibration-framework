# Advanced Bayesian Calibration Framework

A modular Python framework for the Bayesian calibration of complex
simulation models. It brings together, in one workflow:

* **sensitivity screening**: Morris elementary effects (run through SALib
  or Dakota) and Spearman correlation analysis with false-discovery-rate
  control, between inputs and outputs and between the outputs of
  different experiments;
* **Gaussian Process surrogates** with a configurable kernel (RBF,
  Matern, rational quadratic, mixed (sums and products));
* **Principal Component Analysis** of time-dependent outputs, so that a
  quantity of interest (QoI) measured over time is calibrated in a
  reduced space;
* **standard Bayesian calibration** with the affine-invariant ensemble
  sampler `emcee`;
* **hierarchical Bayesian calibration** with a Metropolis-Hastings-within-
  Gibbs sampler, which separates the population distribution of the
  parameters from experiment-to-experiment variability;
* **convergence diagnostics** (R-hat, ESS, MCSE/sigma, acceptance,
  wall-clock time, adaptation health) and **predictive validation**
  (z-scores, coverage, RMSE).

The framework is solver-agnostic: it reads the results of simulations in
a documented format and needs no assumption on the physics of the model.
It was developed and demonstrated in the PhD thesis of Sara Maccario
(EPFL) on nuclear fuel performance models.

> **Status:** private repository, under review before public release.

## Credits

The hierarchical MH-within-Gibbs sampler (`src/abcf/calibration/hierarchical/mhwg/`)
was written by **Gustav Robertson**; see Robertson et al. (2026) [DOI:10.1016/j.nucengdes.2026.115030]. The changes made for this framework are
listed in the header of each file and do not alter the algorithm.

The integration into a complete framework (sensitivity screening, GP and
PCA surrogates, standard calibration, driver of the hierarchical sampler, diagnostics, validation) is the work of **Sara Maccario**.

## Installation

```bash
git clone https://github.com/saramac25/advanced-bayesian-calibration-framework.git
cd advanced-bayesian-calibration-framework
conda env create -f environment.yml      # or: pip install -e .
conda activate abcf
```

Python >= 3.9. Main dependencies: NumPy, SciPy, pandas, scikit-learn,
h5py, emcee, ArviZ, SALib, statsmodels, matplotlib.

## Workflow

| Step | Script | What it does | Output (in `results/`) |
|---|---|---|---|
| 01 | `01_sensitivity_morris.py` | Morris screening with the model itself (SALib + your `ModelInterface`, or a Dakota run) | `sensitivity/` |
| 02 | `02_correlation_analysis.py` | Spearman input-output and output-output correlation (BH-FDR), redundancy groups | `correlation/` |
| 03 | `03_train_surrogates.py` | GPs per case (PCA for a time-dependent QoI), test on held-out runs | `surrogates/` |
| 04 | `04_run_standard.py` | standard calibration (emcee), walkers and steps tuned | `calibration/SC_*.h5` |
| 05 | `05_run_hierarchical.py` | hierarchical calibration (MH-within-Gibbs), several chains | `calibration/HC_*.h5` |
| 06 | `06_diagnostics.py` | R-hat, ESS, MCSE/sigma, acceptance, wall clock, adaptation collapse | `diagnostics/` |
| 07 | `07_postprocessing.py` | posterior summaries and plots: standard, hierarchical or both | `postprocessing/` |
| 08 | `08_validation.py` | z-scores, coverage, RMSE from solver re-runs with posterior samples | `validation/` |

Each script is run with the configuration file:

```bash
python workflows/05_run_hierarchical.py my_study/config.yaml
```

The same scripts handle a **scalar** and a **time-dependent** QoI: a
scalar QoI is a time series with a single time point, and the PCA step
is skipped.

## Your data

1. Sample the parameters and run your solver for every experiment (with
   Dakota, a Python driver or any other tool).
2. Store the runs and the measurements in `input_data.h5`, as described
   in [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md). For a scalar QoI,
   `tools/csv_to_input_h5.py` builds the file from CSV tables.
3. Copy `workflows/config_template.yaml` next to your data and edit it;
   every option is explained in [`docs/CONFIG.md`](docs/CONFIG.md).

To let the framework run your model (Morris screening), implement the
three methods of `abcf.model_interface.ModelInterface`: write the input
file, execute the solver, read the QoI.

## Example

An analytic toy model with six "experiments" exercises the whole
workflow in a few minutes:

```bash
python examples/toy_model.py                 # writes the example data
python examples/run_all.py toy_scalar        # or toy_time_dependent
```

## Citing

If you use this framework, please cite the PhD thesis and Robertson et
al. (2025); see [`CITATION.cff`](CITATION.cff).

## License

To be defined before the public release.
