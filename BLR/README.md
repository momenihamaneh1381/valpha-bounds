# Bayesian linear regression

This module reproduces Figures 4--5 using synthetic data.

From the repository root, install `requirements.txt` and run:

```bash
python reproduce_figures.py --only blr
```

To redraw directly from the saved summary CSV, run from `BLR/`:

```bash
MPLBACKEND=Agg python paper_figures.py
```

To rerun the Monte Carlo study, run from `BLR/`:

```bash
python execute_notebooks.py BLR.ipynb
```

This executes and saves the notebook, replacing the tables, figures, and
configuration in `outputs/`. `generalization_bounds.py` implements the
estimators; `paper_figures.py` uses the shared `../paper_figure_style.py`.

See [REPRODUCE.md](REPRODUCE.md) for parameters, selection pools, and uncertainty
estimates. Existing `corollary1_*` function and CSV names denote the paper's
Corollary 2 and are retained for saved-data compatibility.
