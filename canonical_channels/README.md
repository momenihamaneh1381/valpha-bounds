# Canonical channels

This Python implementation computes Figure 6 (the combined BSC/BEC/Z-channel
comparison) and Figure 7 (the alpha trade-off) directly from analytical
formulas.

From the repository root, install `requirements.txt` and run:

```bash
python reproduce_figures.py --only canonical
```

The paper PDFs are written to `figures/Canonical-channels.pdf` and
`figures/Trade-off.pdf`. Use `--output-dir PATH` to change the destination.

To export the figures and numerical arrays directly, run from this directory:

```bash
MPLBACKEND=Agg python plot_figures.py
```

This writes the combined channel figure, trade-off figure (PDF/PNG/EPS), and
`channel_curves.npz` to `paper_figures/`. The arrays are recomputed on every
run. NumPy, Matplotlib, and the shared `../paper_figure_style.py` are required.

## Formulas and parameters

Entropy and mutual information use natural logarithms (nats):

```text
H(p) = -p*log(p) - (1-p)*log(1-p), with H(0)=H(1)=0
BSC: I = log(2) - H(p)
BEC: I = (1-epsilon)*log(2)
Z:   I = H((1-p)*q) - q*H(p)
```

The comparison panels use uniform input (`q=0.5`) and omit the common factor
`sqrt(sigma**2/n)=1/2`, plotting `V_2` against `sqrt(2*I)` on a 1001-point
channel grid over `[0,1]`.

The trade-off shows full bounds with `q=0.8`, `p=0.3`, `sigma**2=1/4`, `n=1`,
and 1000 alpha values from `1.01` to `20`. Source-PDF labels and ticks are
10 pt, panel titles 12 pt, and legends 8 pt.
