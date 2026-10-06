"""Step 01 - Morris screening of the model parameters.

Runs the model directly (no surrogate) to rank the parameters by their
elementary effects, before the training samples are generated.

  SALib backend : python workflows/01_sensitivity_morris.py config.yaml
                  (needs sensitivity.morris.model = "file.py:ClassName",
                   a ModelInterface subclass, and parameters.<name>.bounds)
  Dakota backend: run Dakota with workflows/templates/dakota_morris.in, then
                  python workflows/01_sensitivity_morris.py config.yaml --dakota-tabular dakota_morris.dat

Output: results/sensitivity/morris_<case>.csv (mu*, mu, sigma per parameter).
"""

import importlib.util
from pathlib import Path

from abcf.config import load_config, output_dir, resolve_path
from abcf.sensitivity.morris import run_morris_dakota, run_morris_salib
from abcf.workflow import parse_args


def load_model(cfg):
    spec = cfg["sensitivity"]["morris"]["model"]
    if not spec or ":" not in spec:
        raise ValueError('Set sensitivity.morris.model to "my_solver.py:MySolver".')
    file, cls = spec.rsplit(":", 1)
    path = resolve_path(cfg, file)
    module_spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, cls)()


def main():
    args = parse_args(__doc__, lambda p: p.add_argument(
        "--dakota-tabular", default=None, help="Dakota tabular file of a psuade_moat run"))
    cfg = load_config(args.config)
    m = cfg["sensitivity"]["morris"]
    bounds = {n: s["bounds"] for n, s in cfg["parameters"].items() if s.get("bounds") is not None}
    if len(bounds) != len(cfg["parameters"]) or not bounds:
        raise ValueError("Morris needs parameters.<name>.bounds for every parameter.")
    out = output_dir(cfg, "sensitivity")

    if args.dakota_tabular or m["backend"] == "dakota":
        if not args.dakota_tabular:
            raise ValueError("Dakota backend: pass --dakota-tabular <file>.")
        table, _, _ = run_morris_dakota(resolve_path(cfg, args.dakota_tabular), bounds,
                                        m["dakota_response_names"])
        name = Path(args.dakota_tabular).stem
        table.to_csv(out / f"morris_{name}.csv", index=False)
        print(table.to_string(index=False))
        return

    model = load_model(cfg)
    cases = m["cases"]
    if cases == "all":
        from abcf.io import load_input_data
        cases = load_input_data(resolve_path(cfg, cfg["data"]["input_file"])).case_ids
    for cid in cases:
        table, design, Y = run_morris_salib(model, cid, bounds, m, resolve_path(cfg, m["run_dir"]),
                                            seed=int(cfg["project"]["seed"]))
        table.to_csv(out / f"morris_{cid}.csv", index=False)
        print(f"\nMorris - case {cid}\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()
