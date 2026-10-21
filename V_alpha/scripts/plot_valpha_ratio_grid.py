import argparse
import os
import pickle

import matplotlib.pyplot as plt

from scripts.plot_valpha_ratio_to_error import (
    # Compatibility: old results.pkl files may refer to
    # scripts.plot_valpha_ratio_grid.NestedDict or __main__.NestedDict.
    # Keep this name available while unpickling, including under python -m.
    NestedDict,
    LABELS,
    MARKERS,
    collect_ratios,
    validate_results_for_methods,
)


METHODS = [
    "theorem8_samplewise_loss_difference_valpha_bound",
    "theorem9_samplewise_single_loss_valpha_bound",
    "theorem10_samplewise_loss_pair_valpha_bound",
    "theorem12_samplewise_binarized_loss_difference_valpha_bound",
    "theorem11_samplewise_binarized_single_loss_valpha_bound",
    "theorem13_samplewise_binarized_loss_pair_valpha_bound",
]

LABELS.setdefault("theorem11_samplewise_binarized_single_loss_valpha_bound", r"T11: bin. single loss")
LABELS.setdefault("theorem13_samplewise_binarized_loss_pair_valpha_bound", r"T13: bin. loss pair")
MARKERS.setdefault("theorem11_samplewise_binarized_single_loss_valpha_bound", "v")
MARKERS.setdefault("theorem13_samplewise_binarized_loss_pair_valpha_bound", "<")


def main():
    parser = argparse.ArgumentParser(description="Plot a grid of V_alpha ratio-to-error experiments.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--ns", nargs="+", type=int, required=True)
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    rows = len(args.inputs)
    fig, axes = plt.subplots(rows, 2, figsize=(9.2, 3.0 * rows), squeeze=False, constrained_layout=True)

    for row, (input_path, label) in enumerate(zip(args.inputs, args.labels)):
        with open(input_path, "rb") as f:
            results = pickle.load(f)
        validate_results_for_methods(results, args.ns, args.epoch, METHODS)
        ratios_01 = collect_ratios(results, args.ns, args.epoch, "01", METHODS)
        ratios_ce = collect_ratios(results, args.ns, args.epoch, "ce", METHODS)

        # plot_one_axis uses the module-level method list, so draw explicitly here.
        for col, (loss_type, ratios) in enumerate((("01", ratios_01), ("ce", ratios_ce))):
            ax = axes[row, col]
            ax.axhline(1.0, linestyle="--", linewidth=1.2, color="black", label="Error ratio = 1")
            for method in METHODS:
                ax.errorbar(
                    args.ns,
                    ratios[method]["mean"],
                    yerr=ratios[method]["sem"],
                    marker=MARKERS.get(method, "o"),
                    linewidth=1.6,
                    capsize=3,
                    label=LABELS.get(method, method),
                )
            ax.set_xscale("log")
            ax.set_xticks(args.ns)
            ax.set_xticklabels([str(n) for n in args.ns])
            ax.set_xlabel(r"$n$")
            ax.set_ylabel(r"bound / $|$empirical gap$|$")
            ax.set_title((r"0--1 loss" if loss_type == "01" else "clipped cross-entropy") + f" ({label})")
            ax.grid(True, which="both", alpha=0.25)
        axes[row, 1].legend(loc="best", fontsize=7)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")
    eps_output = os.path.splitext(args.output)[0] + ".eps"
    fig.savefig(eps_output, format="eps", bbox_inches="tight")
    print(f"Wrote {eps_output}")


if __name__ == "__main__":
    main()
