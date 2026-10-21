import argparse
import os
import pickle
import sys
import types

import matplotlib.pyplot as plt
import numpy as np


class NestedDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


# Compatibility for old pickled results.
parse_results_module = types.ModuleType("scripts.parse_results")
parse_results_module.NestedDict = NestedDict
sys.modules.setdefault("scripts.parse_results", parse_results_module)


MAIN_METHODS = [
    "theorem8_samplewise_loss_difference_valpha_bound",
    "theorem9_samplewise_single_loss_valpha_bound",
    "theorem10_samplewise_loss_pair_valpha_bound",
    # "theorem11_samplewise_binarized_single_loss_valpha_bound",
    "theorem12_samplewise_binarized_loss_difference_valpha_bound",
    # "theorem13_samplewise_binarized_loss_pair_valpha_bound",
]

LABELS = {
    "theorem8_samplewise_loss_difference_valpha_bound": r"T8: loss diff.",
    "theorem9_samplewise_single_loss_valpha_bound": r"T9: single loss",
    "theorem10_samplewise_loss_pair_valpha_bound": r"T10: loss pair",
    # "theorem11_samplewise_binarized_single_loss_valpha_bound": r"T11: bin. single loss",
    "theorem12_samplewise_binarized_loss_difference_valpha_bound": r"T12: bin. loss diff.",
    # "theorem13_samplewise_binarized_loss_pair_valpha_bound": r"T13: bin. loss pair",
}

MARKERS = {
    "theorem8_samplewise_loss_difference_valpha_bound": "o",
    "theorem9_samplewise_single_loss_valpha_bound": "s",
    "theorem10_samplewise_loss_pair_valpha_bound": "^",
    # "theorem11_samplewise_binarized_single_loss_valpha_bound": "v",
    "theorem12_samplewise_binarized_loss_difference_valpha_bound": "D",
    # "theorem13_samplewise_binarized_loss_pair_valpha_bound": "<",
}


def latest_epoch(results, n, requested_epoch):
    if requested_epoch is not None:
        if requested_epoch not in results[n]:
            raise KeyError(
                f"epoch {requested_epoch} not found for n={n}; "
                f"available={sorted(results[n].keys())}"
            )
        return requested_epoch
    return sorted(results[n].keys())[-1]


def mean_sem(xs):
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return np.nan, np.nan
    if xs.size == 1:
        return float(xs[0]), 0.0
    return float(np.mean(xs)), float(np.std(xs, ddof=1) / np.sqrt(xs.size))


def signed_empirical_gap(result, loss_type):
    train, ghost = result[f"loss_{loss_type}"]
    return float(ghost - train)


def empirical_gap_magnitude(result, loss_type):
    # The paper bounds |gen|, so ratios use the magnitude of the empirical
    # selected-versus-ghost gap rather than its sign.
    return abs(signed_empirical_gap(result, loss_type))


def method_bound(result, method, loss_type):
    proxy = result.get(method)
    if proxy is None:
        raise KeyError(
            f"missing required method {method!r}; re-run scripts.parse_results "
            "with the theorem flags used by the reproduction scripts"
        )
    if method.startswith("theorem") and not proxy.get("coordinatewise", False):
        raise ValueError(
            f"{method} is present but is not marked coordinatewise=True. "
            "This is a stale pre-C1 pooled result; re-parse the saved checkpoints."
        )
    key = f"best_gap_bound_{loss_type}"
    if key not in proxy:
        raise KeyError(f"{method} is missing required result field {key!r}")
    return float(proxy[key])


def validate_results_for_methods(results, ns, epoch, methods, gap_epsilon=1e-12):
    for n in ns:
        ep = latest_epoch(results, n, epoch)
        seed_results = results[n][ep]
        if not seed_results:
            raise ValueError(f"no seed results found for n={n}, epoch={ep}")
        for seed_index, result in enumerate(seed_results):
            for method in methods:
                for loss_type in ("01", "ce"):
                    method_bound(result, method, loss_type)
            for loss_type in ("01", "ce"):
                gap = empirical_gap_magnitude(result, loss_type)
                if not np.isfinite(gap):
                    raise ValueError(
                        f"non-finite empirical gap for n={n}, seed_index={seed_index}, "
                        f"loss={loss_type}"
                    )
                if gap <= gap_epsilon:
                    raise ValueError(
                        f"|empirical gap|={gap:.3g} is <= gap_epsilon={gap_epsilon:.3g} "
                        f"for n={n}, seed_index={seed_index}, loss={loss_type}; "
                        "the bound/|gap| ratio is undefined. Report the bound and "
                        "gap separately or choose a pre-declared larger-scale summary."
                    )


def collect_ratios(results, ns, epoch, loss_type, methods, gap_epsilon=1e-12):
    out = {method: {"mean": [], "sem": []} for method in methods}

    for n in ns:
        ep = latest_epoch(results, n, epoch)
        seed_results = results[n][ep]

        for method in methods:
            ratios = []
            for result in seed_results:
                gap = empirical_gap_magnitude(result, loss_type)
                bound = method_bound(result, method, loss_type)
                if not np.isfinite(gap) or not np.isfinite(bound):
                    raise ValueError(
                        f"non-finite bound or gap for n={n}, method={method}, "
                        f"loss={loss_type}"
                    )
                if gap <= gap_epsilon:
                    raise ValueError(
                        f"|empirical gap|={gap:.3g} is <= gap_epsilon={gap_epsilon:.3g} "
                        f"for n={n}, method={method}, loss={loss_type}"
                    )
                ratios.append(bound / gap)

            m, s = mean_sem(ratios)
            out[method]["mean"].append(m)
            out[method]["sem"].append(s)

    return out


def plot_one_axis(ax, ns, ratios, loss_type, use_log):
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="black", label="Error ratio = 1")

    for method in MAIN_METHODS:
        y = np.asarray(ratios[method]["mean"], dtype=float)
        yerr = np.asarray(ratios[method]["sem"], dtype=float)

        ax.errorbar(
            ns,
            y,
            yerr=yerr,
            marker=MARKERS.get(method, "o"),
            linewidth=1.8,
            capsize=3,
            label=LABELS.get(method, method),
        )

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"bound / $|$empirical gap$|$")

    if use_log:
        ax.set_yscale("log")

    if loss_type == "01":
        ax.set_title(r"0--1 loss")
    else:
        ax.set_title(r"clipped cross-entropy")

    ax.grid(True, which="both", alpha=0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/mnist-4vs9-CNN-valpha-main/results.pkl",
        help="path to parsed results.pkl",
    )
    parser.add_argument(
        "--output",
        default="../figures/mnist-4vs9-CNN-valpha-main-ratio-to-error.pdf",
        help="output PDF path; default inferred from input",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=200,
        help="epoch to plot",
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=None,
        help="sample sizes to plot; default all available",
    )
    parser.add_argument(
        "--include_t11",
        action="store_true",
        help="also plot Theorem 11. Usually omitted because T11/T12/T13 overlap.",
    )
    parser.add_argument(
        "--include_t13",
        action="store_true",
        help="also plot Theorem 13. Usually omitted because T11/T12/T13 overlap.",
    )
    parser.add_argument(
        "--linear_ce",
        action="store_true",
        help="use linear y-axis for CE instead of log scale",
    )
    parser.add_argument(
        "--gap_epsilon",
        type=float,
        default=1e-12,
        help=(
            "hard-error threshold for |ghost-train|; ratios are undefined at "
            "or below this value"
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="optional figure title",
    )
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        results = pickle.load(f)

    ns = args.ns if args.ns is not None else sorted(results.keys())

    global MAIN_METHODS
    methods = list(MAIN_METHODS)
    if args.include_t11:
        methods.append("theorem11_samplewise_binarized_single_loss_valpha_bound")
        LABELS["theorem11_samplewise_binarized_single_loss_valpha_bound"] = r"T11: bin. single loss"
        MARKERS["theorem11_samplewise_binarized_single_loss_valpha_bound"] = "v"
    if args.include_t13:
        methods.append("theorem13_samplewise_binarized_loss_pair_valpha_bound")
        LABELS["theorem13_samplewise_binarized_loss_pair_valpha_bound"] = r"T13: bin. loss pair"
        MARKERS["theorem13_samplewise_binarized_loss_pair_valpha_bound"] = "<"
    MAIN_METHODS = methods

    validate_results_for_methods(results, ns, args.epoch, MAIN_METHODS, args.gap_epsilon)
    ratios_01 = collect_ratios(results, ns, args.epoch, "01", MAIN_METHODS, args.gap_epsilon)
    ratios_ce = collect_ratios(results, ns, args.epoch, "ce", MAIN_METHODS, args.gap_epsilon)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)

    plot_one_axis(axes[0], ns, ratios_01, "01", use_log=False)
    plot_one_axis(axes[1], ns, ratios_ce, "ce", use_log=not args.linear_ce)

    axes[1].legend(loc="best", fontsize=8)

    if args.title:
        fig.suptitle(args.title)

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = base + "_ratio_to_error.pdf"

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")

    eps_output = os.path.splitext(args.output)[0] + ".eps"
    fig.savefig(eps_output, format="eps", bbox_inches="tight")
    print(f"Wrote {eps_output}")


if __name__ == "__main__":
    main()