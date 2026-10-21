# Generalization bounds via V_alpha

Python code for the paper's deep-learning, Bayesian linear regression (BLR),
and canonical-channel experiments.

## Reproduce the paper figures

Use Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_figures.py
```

This generates all seven paper PDFs in `figures/` using the included numerical
results and analytical channel formulas. No training or dataset download is
needed. Use `--only deep-learning`, `--only blr`, or `--only canonical` to select
an experiment family; `--output-dir PATH` changes the destination.

| Figures | Experiments | Instructions |
| --- | --- | --- |
| 1--3 | MNIST, CIFAR-10, noisy MNIST | [V_alpha](V_alpha/README.md) |
| 4--5 | BLR bounds and alpha selection | [BLR](BLR/README.md) |
| 6--7 | BSC/BEC/Z-channel and alpha trade-off | [Canonical channels](canonical_channels/README.md) |

## Run the experiments

The same environment supports rerunning the experiments.

Each module's instructions describe recomputation. BLR generates synthetic
data; the channels use analytical formulas. MNIST/CIFAR training scripts
download their datasets and required pretrained weights. Trained checkpoints
are not included; checkpoint re-evaluation requires the original saved runs or
new training runs. See [the deep-learning protocol](V_alpha/REPRODUCIBILITY.md).

## Attribution

The deep-learning pipeline extends
[BinaryJS](https://github.com/Yuxin-Dong/BinaryJS), associated with Dong et al.,
*Exactly Tight Information-theoretic Generalization Bounds via Binary
Jensen-Shannon Divergence* (ICML 2025).
The vendored nnlib library is Copyright (c) 2020 Hrayr Harutyunyan; its
[MIT license](V_alpha/nnlib/LICENSE) is retained.
