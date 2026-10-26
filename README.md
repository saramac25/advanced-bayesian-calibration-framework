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

## Credits

**Hierarchical sampler.** The MH-within-Gibbs building blocks in
`src/abcf/calibration/hierarchical/mhwg/` were written by **Gustav Robertson**
(Uppsala University) during his PhD and shared with Sara Maccario for her doctoral
work. The algorithm is described in:

> G. Robertson, H. Sjöstrand, P. Blair, P. Andersson, A. Toptan, S. Maccario,
> M. Hursin (2026). Credible fission gas release calibration: A hierarchical
> Metropolis–Hastings-within-Gibbs approach to address model inadequacy.
> *Nuclear Engineering and Design* 457, 115030.
> [doi:10.1016/j.nucengdes.2026.115030](https://doi.org/10.1016/j.nucengdes.2026.115030)

The changes made for this framework are listed in the header of each file and do
not alter the algorithm. The structure of the GP ensemble in
`src/abcf/surrogate/gp.py` is also derived from his code.

**Framework.** The integration into a complete, solver-agnostic framework
(sensitivity screening, GP and PCA surrogates, standard calibration, driver of the
hierarchical sampler and joint shift move, diagnostics, validation, examples) is the
work of **Sara Maccario** (PSI / EPFL), under the supervision of Mathieu Hursin.

## Acknowledgements

* The TRANSURANUS calculations, their post-processed results and the set-up of the
  time-dependent IFPE/Halden case (`examples/thesis_ch5_transuranus/`) were provided
  by Gustav Robertson. Neither the IFPE measurements (OECD/NEA) nor the TRANSURANUS
  results are included in this repository: the example runs on dummy data generated
  by `make_facsimile.py`.
* The rod power histories used to generate the synthetic OFFBEAT experiments
  (`examples/thesis_ch5_offbeat/`) were provided by Daniele Timpano; they are not
  included in this repository.
* Parts of the code were restructured with the help of an AI assistant (Claude,
  Anthropic). All changes were reviewed and tested by the author.

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
| 02 | `02_correlation_analysis.py` | Spearman input-output and output-output correlation (BH-FDR), redundancy groups, optional validation split | `correlation/` |
| 03 | `03_train_surrogates.py` | GPs per case (PCA for a time-dependent QoI), test on test runs; optional kernel comparison | `surrogates/` |
| 04 | `04_run_standard.py` | standard calibration (emcee), walkers and steps tuned | `calibration/SC_*.h5` |
| 05 | `05_run_hierarchical.py` | hierarchical calibration (MH-within-Gibbs), several chains | `calibration/HC_*.h5` |
| 06 | `06_diagnostics.py` | R-hat, ESS, MCSE/sigma, acceptance, wall clock, adaptation collapse | `diagnostics/` |
| 07 | `07_postprocessing.py` | posterior summaries and plots: standard, hierarchical or both | `postprocessing/` |
| 08 | `08_validation.py` | z-scores, coverage, RMSE from solver re-runs with posterior samples, or chi2_E / chi2_tot through the GPs of the validation cases | `validation/` |

Run the whole chain with one command (main entry point):

```bash
python workflows/run_workflow.py my_study/config.yaml              # all applicable steps
python workflows/run_workflow.py my_study/config.yaml --steps 3-7  # only some steps
```

or each step on its own, e.g. after changing the configuration:

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

## Examples

An analytic toy model with six "experiments" exercises the whole
workflow in a few minutes:

```bash
python examples/toy_model.py                 # writes the example data
python examples/run_all.py toy_scalar        # or toy_time_dependent
```

`examples/thesis_ch5_offbeat/` contains the scalar case studies of
Chapter 5 of the thesis (130 fuel rods simulated with OFFBEAT, synthetic
experiments with known true parameters: epistemic uncertainty, aleatoric
uncertainty, model bias), with their data and one script that runs them
and draws the figures; see its README.

## Citing

If you use this framework, please cite the PhD thesis and Robertson et
al. (2026); see [`CITATION.cff`](CITATION.cff).

## License

BSD 3-Clause License, see [`LICENSE`](LICENSE). The files in `src/abcf/calibration/hierarchical/mhwg/` are by Gustav Robertson and are published here as agreed with him during his PhD.
