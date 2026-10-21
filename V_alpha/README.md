# MNIST and CIFAR-10 experiments

This module produces Figure 1 (MNIST 4-vs-9), Figure 2 (CIFAR-10 with
ImageNet-pretrained ResNet-50), and Figure 3 (MNIST with label-flip
probabilities `0.05`, `0.10`, and `0.15`).

From the repository root, install `requirements.txt` and run:

```bash
python reproduce_figures.py --only deep-learning
```

The five included `results/*/results.pkl` aggregates suffice for redraw.
Training checkpoints and per-model saved data are not included.

For training or checkpoint re-evaluation, follow
[REPRODUCIBILITY.md](REPRODUCIBILITY.md),
which specifies preprocessing, seeds, losses, estimators, and commands.

| Path | Purpose |
| --- | --- |
| `configs/` | Model architectures |
| `scripts/train_classifier.py` | Training and supersample construction |
| `scripts/parse_results.py` | Checkpoint evaluation and bound estimates |
| `modules/bound_utils.py` | Bound and dependence estimators |
| `reproduce/` | Figure-specific entry points |
| `nnlib/` | Vendored training utilities; see its `LICENSE` |

BLR and canonical-channel experiments are in the sibling modules.
