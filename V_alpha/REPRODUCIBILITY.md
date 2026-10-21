# Deep-learning experiment protocol

Use the Python environment described in the root [README](../README.md).
Run the commands below from `V_alpha/` with that environment active:

```bash
export PYTHON_BIN="$(command -v python)"
```

## Redraw or re-evaluate

Redraw the included numerical aggregates without training:

```bash
bash reproduce/figures_from_aggregates.sh
```

To re-evaluate saved checkpoints instead:

```bash
bash reproduce/figure1_mnist.sh
bash reproduce/figure2_cifar10.sh
bash reproduce/figure3_mnist_noise.sh
```

These require every run's `saved_data.pkl` and final checkpoint:

```text
results/<experiment>/n=<n>,seed=<seed>,S_seed=<selector>/saved_data.pkl
results/<experiment>/n=<n>,seed=<seed>,S_seed=<selector>/checkpoints/epoch199.mdl
```

CIFAR-10 uses `epoch39.mdl` instead of `epoch199.mdl`. Its saved data must
include `all_examples_wo_data_aug`: the same ordered `2n` supersample images
and labels with augmentation disabled. The scripts stop on missing artifacts.
Set `RESULTS_DIR` and `FIGURES_DIR` for alternative input/output locations;
`PARSE_DEVICE` defaults to `cpu`. `SKIP_PARSE=1` redraws existing aggregates
instead of loading checkpoints. Load pickle/checkpoint files only from
trusted sources.

## Training from scratch

The loaders download MNIST/CIFAR-10 and the required pretrained weights.
Training scripts default to CUDA; set `DEVICE=cpu` for CPU execution.

```bash
bash scripts/run_valpha_mnist_main.sh
bash scripts/run_valpha_mnist_noise_005_main.sh
bash scripts/run_valpha_mnist_noise_01_main.sh
bash scripts/run_valpha_mnist_noise_015_main.sh
bash scripts/run_valpha_cifar10_pretrained_resnet50_main.sh
```

| Setting | MNIST, including noisy variants | CIFAR-10 |
| --- | --- | --- |
| Architecture | 4-convolution CNN | ResNet-50, `IMAGENET1K_V1`, new 10-way head; all layers fine-tuned |
| Sample sizes `n` | 75, 250, 1000, 4000 | 1000, 5000, 20000 |
| Outer seeds `seed` | 0--4 | 0--1 |
| Selector seeds `S_seed` | 0--29 | 0--39 |
| Epochs / batch size | 200 / 128 | 40 / 64 |
| Optimizer | Adam, LR 0.001 | SGD, LR 0.01, momentum 0.9 |

Both use constant learning rates and weight decay zero. Each MNIST setting
trains 600 models (2400 including noise); CIFAR-10 trains 240 models.
`seed` determines the supersample, and `S_seed` determines the selector mask.
MNIST enables deterministic training with RNG seed 42 and zero data-loader
workers. CIFAR training does not enable deterministic mode, so identical
weights are not guaranteed on a rerun.

## Preprocessing

MNIST retains digits 4 and 9 with binary labels, converts images to tensors,
and normalizes with mean `0.456` and standard deviation `0.224`, without
resizing or augmentation. Noisy variants apply binary label flips with the
specified probability; the corrupted supersample is fixed across selectors.

CIFAR-10 first makes a seeded 80/20 split of its training partition, then
samples `2n` images without replacement from the training subset. Training
applies horizontal flips and random 32-by-32 crops with padding 4. Images are
converted to tensors, normalized with means `(0.485,0.456,0.406)` and standard
deviations `(0.229,0.224,0.225)`, then bilinearly resized to 224-by-224.
Evaluation uses the same normalization and resize with augmentation disabled.

## Losses and plotted quantities

Training uses unclipped cross-entropy. Evaluation uses 0-1 loss and
cross-entropy clipped to `[0,1]`:

```text
0-1 loss = 1[argmax(logits) != y]
clipped CE = min(cross_entropy(logits, y), 1)
signed gap = ghost_loss - train_loss
ratio per outer seed = estimated bound / abs(signed gap)
```

The figure averages these ratios and reports SEM across outer seeds: five
for MNIST and two for CIFAR-10. A gap magnitude at or below `1e-12` raises
an error. Numerical summaries retain signed gaps separately.

## Estimators

`scripts.parse_results` computes the information baselines and coordinatewise
T8--T13 estimates. For a fixed supersample and coordinate `i`, it estimates
the conditional law using selector draws for that coordinate and the exact
fair-selector prior `P(U_i=0)=P(U_i=1)=1/2`. It forms local contributions before
averaging coordinates and outer seeds.

T8 and T10 use their conditional statements; T9 and T11 use the two-sided
conditional corollary with `symmetrization=average`. T12 and T13 use its
conditional binarized difference/pair statements. The paper scripts use the
alpha grid `1.005, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0`. For a fair binary selector,
the coordinatewise dependence and weighted quantities are alpha-invariant;
the first grid value is a representative, not a tuned optimum.

Continuous clipped-CE statistics use 30 bins with 30 selector draws per MNIST
coordinate or 40 per CIFAR coordinate. These finite-sample plug-in estimates
have sampling/discretization error and are not confidence bounds for the
unbinned population quantities.

## Numerical summaries

```bash
python -m scripts.compare_theorem_bounds_numeric \
  --input results/mnist-4vs9-CNN-valpha-main/results.pkl --epoch 200 --loss_type ce
```

To record the current software/hardware environment, run
`python -m scripts.report_environment --output environment.json`.
