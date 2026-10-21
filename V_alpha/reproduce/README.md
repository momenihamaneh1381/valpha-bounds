# Figure entry points

From `V_alpha/`, with the root Python environment active:

```bash
export PYTHON_BIN="$(command -v python)"
bash reproduce/figures_from_aggregates.sh
```

This redraws Figures 1--3 from the included aggregates. The individual
`figure1_mnist.sh`, `figure2_cifar10.sh`, and `figure3_mnist_noise.sh` scripts
re-evaluate checkpoints by default; `SKIP_PARSE=1` selects aggregate redraw.

`RESULTS_DIR` selects inputs, `FIGURES_DIR` selects outputs, and `PARSE_DEVICE`
selects the checkpoint inference device (default `cpu`). Required artifacts
and training commands are documented in
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
