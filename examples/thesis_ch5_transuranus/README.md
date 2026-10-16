# Thesis, Chapter 5: TRANSURANUS / Halden time-dependent case study

Standard (emcee) and hierarchical (MH-within-Gibbs) calibration of the fuel centreline
temperature histories measured by the thermocouples of Halden fuel rods (IFPE database,
OECD/NEA), simulated with TRANSURANUS. Same pipeline, settings and criteria as the OFFBEAT
synthetic case studies (`examples/thesis_ch5_offbeat`); here the data are real, so there
are no true values and aleatoric variability and model bias cannot be separated.

**The IFPE measurements are not redistributable.** They are read from the user's own
`results.h5` (TRANSURANUS-Dakota runs) and written to `data/input_data.h5`; `data/` and
`results/` are never committed. `make_facsimile.py` writes a file with the same layout and
dummy numbers, to run the whole pipeline without the IFPE data (`--facsimile`).

| | |
|---|---|
| Cases | one per thermocouple (rod and axial node; `node-mean` = rod average), 38 in total |
| Points | steady state (`state == 0`), measurement >= 400 K, not flagged in `filter` (`prepare_data.py`, as in the previous `templib_TU.case._filter`) |
| Parameters | 5 TRANSURANUS multipliers: `cal_gapcon_f` [0.8, 1.2], `cal_lambda_a1`, `cal_lambda_b0`, `cal_lambda_b2`, `cal_delg_f` [0.5, 1.5]; 300 Dakota runs |
| Output reduction | PCA per case, components up to 99% of the variance (as before) |
| Surrogates | one GP per component and case; kernel chosen by the comparison of 5 kernels (stage `kernels`) |
| Measurement uncertainty | 1% of the measured temperature (as in the synthetic cases) |
| Validation | 6 cases with the lowest maximum correlation with the other cases (step 02), through their GPs (chi2_E, chi2_tot, coverage) + 93 posterior samples to re-run TRANSURANUS (Wilks 95/95) |
| Calibrated parameters | `config_tu.yaml`: the parameters influential (Spearman, BH-FDR, rho^2 >= 0.01) in all cases (stage `correlation`): lambda_a1, lambda_b0. Sobol' indices (stage `sobol`, optional) are not used for the choice: unlike the OFFBEAT models, no literature uncertainty is available for the individual TRANSURANUS coefficients, and the indices would only reflect the ranges of the training runs |

## Run

```bash
python run_thesis_tu.py all --facsimile --quick          # pipeline check on dummy data (minutes)
python run_thesis_tu.py prepare --h5 <path>/results.h5   # real data -> data/input_data.h5
python run_thesis_tu.py correlation kernels gp           # common part
#   -> check results/common/correlation/input_output_summary.csv (influential in all cases)
#      and the parameters block of config_tu.yaml
python run_thesis_tu.py calibration                      # steps 04-08 (lengths 10k -> 100k, continued)
python run_thesis_tu.py calibration --hc-steps 150000 --continue-hc   # longer, without restarting
python figures_tu.py --time-label "Time [d]"             # thesis figures, tables, appendix LaTeX
```

| Stage | Output |
|---|---|
| prepare | `data/input_data.h5`, `data/cases_summary.csv` (points kept per case) |
| correlation | `results/common/correlation/` (input-output per case and summary, case-case, `validation_split.csv`) |
| kernels, gp | `results/common/surrogates/` (kernel comparison, GPs, test metrics) |
| sobol | `results/common/sobol/` (S1, ST per case and principal component, variance-weighted indices of the whole history, summary, figure) |
| calibration | `results/tu/` (calibration, diagnostics, postprocessing, validation, 93 posterior samples per method) |
| figures_tu.py | `results/figures_tu/` (Spearman summary, GP parity, population, case-level parameters, prediction + C/E examples; appendix A (PCA, Spearman) and B (prediction + C/E) of every case with their LaTeX; `tu_cases.csv`, `tu_case_metrics.csv`, `tu_case_modes.csv`) |

Source layout of `results.h5` and the filtering: see the docstring of `prepare_data.py`.
